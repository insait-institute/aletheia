"""
bon_livecodebench.py

Best-of-N evaluation on LiveCodeBench using a trained code verifier.

Generates N candidate solutions per problem, uses a verifier to select the
best, then executes the selection on test cases and reports pass@1.

Selection modes:
  list      - all N candidates passed to the verifier in a single prompt
  pairwise  - all C(N,2) pair comparisons, winner chosen by win count

Verifier prompt styles (controlled by --verifier_style):
  thinking  - DeepSeek-R1/lrm style: adds <think>\\n assistant prefix,
              parses answer after </think>
  instruct  - Instruct-tuned style: COT system prompt, no thinking prefix

Verifier prompt types (controlled by --verifier_scoring):
  off  - selection prompt (LIST_REWARD_PROMPT / pair A-or-B)
  on   - scoring prompt (LISTSC_PROMPT scores 0-10 / PAIRSC_PROMPT)

Usage:
  # Full pipeline
  python bon_livecodebench.py \\
    --generator_model /path/to/generator \\
    --verifier_model  /path/to/verifier  \\
    --N 8 --selection_mode list

  # Skip generation (reuse a cached candidates file)
  python bon_livecodebench.py \\
    --verifier_model /path/to/verifier \\
    --N 8 --selection_mode pairwise \\
    --candidates_cache outputs/bon_lcb/candidates_<id>.pkl
"""

import argparse
import ast
import csv
import itertools
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))
from prompts import LIST_REWARD_PROMPT, LIST_REWARD_PROMPT_COT, LISTSC_PROMPT, PAIRSC_PROMPT
from rewards import extract_boxed_contents_list, extract_boxed_contents_score10
from utils import run_inference

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1

_LETTERS = list("ABCDEFGH")

# ─────────────────────────────────────────────────────────────────────────────
# Code generation prompts
# ─────────────────────────────────────────────────────────────────────────────

_CODEGEN_SYSTEM = "You are an expert competitive programmer. Write correct, efficient Python solutions."

_CODEGEN_STDIN = """\
Solve the following programming problem. Write a complete Python solution that reads input from stdin and writes the answer to stdout.

{problem}

Enclose your solution in a Markdown-formatted Python code block:
```python
# solution here
```"""

_CODEGEN_FUNCTIONAL = """\
Solve the following programming problem. Define a class `Solution` with a method `{func_name}`.

{problem}

{starter_section}Enclose your solution in a Markdown-formatted Python code block:
```python
# solution here
```"""


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _problem_meta(problem: dict) -> dict:
    meta = problem.get("metadata", {})
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    platform = (problem.get("platform") or "").lower()
    is_functional = platform == "leetcode"
    return {
        "platform": platform,
        "is_functional": is_functional,
        "func_name": meta.get("func_name", "solve") if is_functional else None,
    }


def load_livecodebench(
    split: str = "test",
    max_problems: int = 0,
    test_field: str = "private_test_cases",
    version_tag: str = "release_v6",
) -> List[dict]:
    log.info(f"Loading LiveCodeBench ({split} / {version_tag})...")
    # version_tag is the HuggingFace dataset configuration name
    try:
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            name=version_tag,
            split=split,
            trust_remote_code=True,
        )
    except Exception:
        # Fall back to loading without a name config (older dataset versions)
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split=split,
            trust_remote_code=True,
        )
    # Only keep problems that have the required test cases
    problems = [p for p in ds if p.get(test_field) or p.get("public_test_cases")]
    # Prefer the requested test_field; mark each problem so downstream knows
    problems = [p for p in problems if p.get(test_field)]
    if max_problems and len(problems) > max_problems:
        problems = problems[:max_problems]
    log.info(f"Loaded {len(problems)} problems with {test_field}")
    return problems


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ─────────────────────────────────────────────────────────────────────────────


def _build_codegen_prompts(problems: List[dict]) -> List[List[dict]]:
    prompts = []
    for p in problems:
        meta = _problem_meta(p)
        content = _strip_html(p.get("question_content", ""))
        if meta["is_functional"]:
            starter = (p.get("starter_code") or "").strip()
            starter_section = f"Starter code:\n```python\n{starter}\n```\n\n" if starter else ""
            user = _CODEGEN_FUNCTIONAL.format(
                func_name=meta["func_name"] or "solve",
                problem=content,
                starter_section=starter_section,
            ).strip()
        else:
            user = _CODEGEN_STDIN.format(problem=content).strip()
        prompts.append(
            [
                {"role": "user", "content": f"{_CODEGEN_SYSTEM}\n\n{user}"},
            ]
        )
    return prompts


def extract_code(text: str) -> str:
    """Extract the last fenced code block; strip think traces first."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    for pat in [r"```python\n(.*?)```", r"```\n?(.*?)```"]:
        matches = re.findall(pat, text, re.DOTALL)
        if matches:
            return matches[-1].strip()
    return text.strip()


def generate_candidates(
    problems: List[dict],
    generator_model: str,
    N: int,
    max_tokens: int = 2048,
    temperature: float = 0.6,
    **kw,
) -> List[List[str]]:
    """Return shape [n_problems][N] list of extracted code strings."""
    prompts = _build_codegen_prompts(problems)
    log.info(f"Generating {N} candidates for {len(problems)} problems...")
    outputs = run_inference(prompts, generator_model, n=N, temperature=temperature, max_tokens=max_tokens, **kw)
    return [[extract_code(ch.text) for ch in req.outputs] for req in outputs]


# ─────────────────────────────────────────────────────────────────────────────
# Verifier prompts
# ─────────────────────────────────────────────────────────────────────────────


def _format_candidates(codes: List[str]) -> Tuple[str, str]:
    """Return (candidate_block_str, valid_options_str)."""
    letters = _LETTERS[: len(codes)]
    parts = [f"[CANDIDATE_{ltr}]\n```python\n{c}\n```\n[/CANDIDATE_{ltr}]" for ltr, c in zip(letters, codes)]
    return "\n\n".join(parts), ", ".join(letters)


def _make_prompt(content: str, style: str, assistant_prefix: str = "<think>\n") -> List[dict]:
    if style == "instruct":
        return [{"role": "user", "content": content}]
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": assistant_prefix},
    ]


def build_list_prompt(problem: dict, candidates: List[str], style: str, scoring: bool) -> List[dict]:
    q = _strip_html(problem.get("question_content", ""))
    cand_str, options = _format_candidates(candidates)
    if scoring:
        content = LISTSC_PROMPT.format(question=q, candidates=cand_str).strip()
        return _make_prompt(content, style)
    if style == "instruct":
        return [
            {"role": "system", "content": LIST_REWARD_PROMPT_COT.format(valid_options=options)},
            {
                "role": "user",
                "content": (
                    f"Here is the coding question followed by the candidate solutions:\n"
                    f"[QUESTION]\n{q}\n[/QUESTION]\n\n{cand_str}\n\n"
                    "Your response should be exactly in the specified format."
                ),
            },
        ]
    content = LIST_REWARD_PROMPT.format(question=q, candidates=cand_str, valid_options=options).strip()
    return _make_prompt(content, style)


def build_pair_prompt(problem: dict, code_a: str, code_b: str, style: str, scoring: bool) -> List[dict]:
    q = _strip_html(problem.get("question_content", ""))
    cand_str = f"[CANDIDATE_A]\n```python\n{code_a}\n```\n[/CANDIDATE_A]\n\n[CANDIDATE_B]\n```python\n{code_b}\n```\n[/CANDIDATE_B]"
    if scoring:
        content = PAIRSC_PROMPT.format(question=q, candidates=cand_str).strip()
        return _make_prompt(content, style)
    if style == "instruct":
        return [
            {"role": "system", "content": LIST_REWARD_PROMPT_COT.format(valid_options="A, B")},
            {
                "role": "user",
                "content": (
                    f"Here is the coding question followed by the candidate solutions:\n"
                    f"[QUESTION]\n{q}\n[/QUESTION]\n\n{cand_str}\n\n"
                    "Your response should be exactly in the specified format."
                ),
            },
        ]
    content = LIST_REWARD_PROMPT.format(question=q, candidates=cand_str, valid_options="A, B").strip()
    return _make_prompt(content, style)


# ─────────────────────────────────────────────────────────────────────────────
# Verdict parsing
# ─────────────────────────────────────────────────────────────────────────────


def _post_think(text: str) -> str:
    return text.split("</think>")[-1].strip() if "</think>" in text else text.strip()


def _parse_list_verdict(text: str, n: int, scoring: bool) -> Optional[int]:
    payload = _post_think(text)
    if scoring:
        scores = extract_boxed_contents_score10(payload)
        if scores and len(scores) == n and scores.count(max(scores)) == 1:
            return scores.index(max(scores))
        return None
    letter = extract_boxed_contents_list(payload)
    if letter and letter in _LETTERS[:n]:
        return _LETTERS.index(letter)
    return None


def _parse_pair_verdict(text: str, scoring: bool) -> Optional[str]:
    """Return 'A' or 'B' (or None)."""
    payload = _post_think(text)
    if scoring:
        scores = extract_boxed_contents_score10(payload)
        if scores and len(scores) == 2 and scores[0] != scores[1]:
            return "A" if scores[0] > scores[1] else "B"
        return None
    letter = extract_boxed_contents_list(payload)
    return letter if letter in ("A", "B") else None


# ─────────────────────────────────────────────────────────────────────────────
# Selection: list mode
# ─────────────────────────────────────────────────────────────────────────────


def select_by_list(
    problems: List[dict],
    candidates_per_problem: List[List[str]],
    verifier_model: str,
    style: str = "thinking",
    scoring: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    **kw,
) -> List[Optional[int]]:
    prompts = [build_list_prompt(p, c, style, scoring) for p, c in zip(problems, candidates_per_problem)]
    log.info(f"List-mode verifier: {len(prompts)} prompts...")
    outputs = run_inference(prompts, verifier_model, n=1, temperature=temperature, max_tokens=max_tokens, **kw)
    selected = []
    for i, (req, cands) in enumerate(zip(outputs, candidates_per_problem)):
        idx = _parse_list_verdict(req.outputs[0].text, len(cands), scoring)
        if idx is None:
            log.warning(f"Problem {i}: unparseable list verdict → counted as fail")
        selected.append(idx)
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Selection: pairwise mode
# ─────────────────────────────────────────────────────────────────────────────


def select_by_pairwise(
    problems: List[dict],
    candidates_per_problem: List[List[str]],
    verifier_model: str,
    style: str = "thinking",
    scoring: bool = False,
    max_tokens: int = 16384,
    temperature: float = 0.6,
    bias_correction: bool = True,
    **kw,
) -> List[Optional[int]]:
    """
    Run all C(N,2) pairwise comparisons (optionally both orderings for position
    bias correction).  Select by win count per problem.
    Returns None for a problem when every pairwise verdict was unparseable.
    """
    all_prompts: List[List[dict]] = []
    # meta: (prob_idx, cand_i, cand_j, ordering)  ordering in {"AB", "BA"}
    meta: List[Tuple[int, int, int, str]] = []

    for prob_idx, (problem, cands) in enumerate(zip(problems, candidates_per_problem)):
        for i, j in itertools.combinations(range(len(cands)), 2):
            all_prompts.append(build_pair_prompt(problem, cands[i], cands[j], style, scoring))
            meta.append((prob_idx, i, j, "AB"))
            if bias_correction:
                all_prompts.append(build_pair_prompt(problem, cands[j], cands[i], style, scoring))
                meta.append((prob_idx, i, j, "BA"))

    log.info(f"Pairwise-mode verifier: {len(all_prompts)} prompts...")
    outputs = run_inference(all_prompts, verifier_model, n=1, temperature=temperature, max_tokens=max_tokens, **kw)

    wins: List[List[int]] = [[0] * len(c) for c in candidates_per_problem]
    for req, (prob_idx, i, j, ordering) in zip(outputs, meta):
        verdict = _parse_pair_verdict(req.outputs[0].text, scoring)
        if verdict is None:
            continue
        # Map A/B verdict back to candidate index given the ordering
        winner = (i if ordering == "AB" else j) if verdict == "A" else (j if ordering == "AB" else i)
        wins[prob_idx][winner] += 1

    selected: List[Optional[int]] = []
    for prob_idx, w in enumerate(wins):
        if max(w) == 0:
            # Every pairwise verdict was unparseable for this problem
            log.warning(f"Problem {prob_idx}: no parseable pairwise verdicts → counted as fail")
            selected.append(None)
        else:
            selected.append(w.index(max(w)))
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Code execution
# ─────────────────────────────────────────────────────────────────────────────


def _run(code: str, stdin: Optional[str], timeout: float) -> Tuple[str, int]:
    """Execute Python code in a temp file; return (stdout, returncode)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(code)
        fname = f.name
    try:
        proc = subprocess.run(
            ["python3", fname],
            input=stdin or "",
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return proc.stdout, proc.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1
    except Exception as e:
        return f"ERROR:{e}", -1
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _functional_harness(solution_code: str, func_name: str, raw_input: str) -> str:
    """Wrap a LeetCode-style solution so we can call it and compare output."""
    try:
        args = [ast.literal_eval(line.strip()) for line in raw_input.strip().splitlines() if line.strip()]
    except Exception:
        args = []
    return f"import sys, json\n{solution_code}\n\n_args = {json.dumps(args)}\n_sol = Solution()\n_res = getattr(_sol, {json.dumps(func_name)})(*_args)\nprint(json.dumps(_res))\n"


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines()).strip()


def _outputs_match(actual: str, expected: str) -> bool:
    a, e = _normalize(actual), _normalize(expected)
    if a == e:
        return True
    try:
        return json.loads(a) == json.loads(e)
    except Exception:
        return False


def run_test_cases(
    code: str,
    problem: dict,
    test_cases: List[dict],
    timeout: float = 10.0,
) -> Tuple[int, int]:
    """Return (n_passed, n_total)."""
    meta = _problem_meta(problem)
    passed = 0
    for tc in test_cases:
        if isinstance(tc, str):
            tc = json.loads(tc)
        inp, expected = tc.get("input", ""), tc.get("output", "")
        if meta["is_functional"]:
            harness = _functional_harness(code, meta["func_name"] or "solve", inp)
            stdout, rc = _run(harness, None, timeout)
        else:
            stdout, rc = _run(code, inp, timeout)
        if rc == 0 and _outputs_match(stdout, expected):
            passed += 1
    return passed, len(test_cases)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


def evaluate_selections(
    problems: List[dict],
    candidates_per_problem: List[List[str]],
    selected_indices: List[Optional[int]],
    test_field: str = "private_test_cases",
    timeout: float = 10.0,
) -> Dict:
    results = []
    for prob_idx, (problem, cands, sel) in enumerate(zip(problems, candidates_per_problem, selected_indices)):
        tcs = problem.get(test_field) or []
        if not tcs:
            continue
        if sel is None:
            # Unparseable verdict: always a fail, no execution needed
            results.append({"prob_idx": prob_idx, "sel": None, "passed": 0, "total": len(tcs), "verdict_fail": True})
            log.info(f"  [{prob_idx}] FAIL (unparseable verdict)")
            continue
        code = cands[min(sel, len(cands) - 1)]
        passed, total = run_test_cases(code, problem, tcs, timeout)
        results.append({"prob_idx": prob_idx, "sel": sel, "passed": passed, "total": total, "verdict_fail": False})
        log.info(f"  [{prob_idx}] cand={sel}  {passed}/{total}")
    pass_at_1 = float(np.mean([r["passed"] == r["total"] for r in results])) if results else 0.0
    return {"pass@1": pass_at_1, "n": len(results), "per_problem": results}


def evaluate_oracle_random(
    problems: List[dict],
    candidates_per_problem: List[List[str]],
    test_field: str = "private_test_cases",
    timeout: float = 10.0,
) -> Dict:
    """Execute every candidate for every problem to compute oracle and random baselines."""
    import random

    oracle_pass, rand_pass = [], []
    for prob_idx, (problem, cands) in enumerate(zip(problems, candidates_per_problem)):
        tcs = problem.get(test_field) or []
        if not tcs:
            continue
        scores = []
        for code in cands:
            passed, total = run_test_cases(code, problem, tcs, timeout)
            scores.append(passed == total)
        oracle_pass.append(any(scores))
        rand_pass.append(scores[random.randrange(len(scores))])
        log.info(f"  [{prob_idx}] oracle={oracle_pass[-1]}  random={rand_pass[-1]}")
    return {
        "oracle_pass@1": float(np.mean(oracle_pass)) if oracle_pass else 0.0,
        "random_pass@1": float(np.mean(rand_pass)) if rand_pass else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main(args):
    out_dir = Path(__file__).parents[0] / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())[:8]
    test_field = "private_test_cases" if args.test_set == "private" else "public_tests"

    # ── Stage 1: Candidate generation ─────────────────────────────────────
    if args.candidates_cache:
        log.info(f"Loading candidates from {args.candidates_cache}")
        with open(args.candidates_cache, "rb") as f:
            cache = pickle.load(f)
        problems: List[dict] = cache["problems"]
        candidates_per_problem: List[List[str]] = cache["candidates"]
    else:
        problems = load_livecodebench(
            split=args.split,
            max_problems=args.max_problems,
            test_field=test_field,
            version_tag=args.lcb_version,
        )
        candidates_per_problem = generate_candidates(
            problems,
            args.generator_model,
            N=args.N,
            max_tokens=args.generator_max_tokens,
            temperature=args.generator_temperature,
            dp_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=args.generator_max_tokens + 4096,
            top_p=0.95,
        )
        cache_path = out_dir / f"candidates_{run_id}.pkl"
        with open(cache_path, "wb") as f:
            pickle.dump({"problems": problems, "candidates": candidates_per_problem, "args": vars(args)}, f)
        log.info(f"Candidates saved → {cache_path}")

    N = len(candidates_per_problem[0])
    log.info(f"{len(problems)} problems × {N} candidates")

    # ── Stage 2: Verifier selection ────────────────────────────────────────
    if args.selections_cache:
        log.info(f"Loading selections from {args.selections_cache}")
        with open(args.selections_cache, "rb") as f:
            sel_cache = pickle.load(f)
        selected_indices: List[int] = sel_cache["selected_indices"]
    elif args.verifier_model:
        common = dict(
            verifier_model=args.verifier_model,
            style=args.verifier_style,
            scoring=args.verifier_scoring,
            max_tokens=args.verifier_max_tokens,
            temperature=0.6,
            dp_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=args.verifier_max_tokens + 4096,
            top_p=0.95,
        )
        if args.selection_mode == "list":
            selected_indices = select_by_list(problems, candidates_per_problem, **common)
        else:
            selected_indices = select_by_pairwise(
                problems,
                candidates_per_problem,
                bias_correction=not args.no_bias_correction,
                **common,
            )
        sel_path = out_dir / f"selections_{args.selection_mode}_{run_id}.pkl"
        with open(sel_path, "wb") as f:
            pickle.dump({"selected_indices": selected_indices, "args": vars(args)}, f)
        log.info(f"Selections saved → {sel_path}")
    else:
        import random

        log.info("No verifier; using random selection baseline")
        selected_indices = [random.randrange(len(c)) for c in candidates_per_problem]

    # ── Stage 3: Execution ─────────────────────────────────────────────────
    log.info("Evaluating verifier selections...")
    verifier_res = evaluate_selections(
        problems,
        candidates_per_problem,
        selected_indices,
        test_field=test_field,
        timeout=args.execution_timeout,
    )

    baselines: Dict = {}
    if args.compute_baselines:
        log.info("Computing oracle + random baselines (executes all candidates)...")
        baselines = evaluate_oracle_random(
            problems,
            candidates_per_problem,
            test_field=test_field,
            timeout=args.execution_timeout,
        )

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(f"  Best-of-{N} | LiveCodeBench | mode={args.selection_mode}")
    print(f"  verifier: {args.verifier_model or 'none'}")
    print("=" * 62)
    if baselines:
        print(f"  random pass@1  : {baselines['random_pass@1']:.4f}  ({baselines['random_pass@1'] * 100:.2f}%)")
        print(f"  oracle pass@1  : {baselines['oracle_pass@1']:.4f}  ({baselines['oracle_pass@1'] * 100:.2f}%)")
    print(f"  verifier pass@1: {verifier_res['pass@1']:.4f}  ({verifier_res['pass@1'] * 100:.2f}%)")
    print(f"  n_problems     : {verifier_res['n']}")
    print("=" * 62 + "\n")

    # Save full results
    results_path = out_dir / f"results_{run_id}.pkl"
    with open(results_path, "wb") as f:
        pickle.dump({"verifier": verifier_res, "baselines": baselines, "args": vars(args), "run_id": run_id}, f)

    # Append CSV summary
    csv_path = out_dir / "bon_results.csv"
    is_new = not csv_path.exists()
    fields = [
        "run_id",
        "generator_model",
        "verifier_model",
        "N",
        "selection_mode",
        "verifier_style",
        "verifier_scoring",
        "n_problems",
        "verifier_pass@1",
        "oracle_pass@1",
        "random_pass@1",
        "split",
        "lcb_version",
        "test_set",
    ]
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if is_new:
            w.writeheader()
        w.writerow(
            {
                "run_id": run_id,
                "generator_model": args.generator_model or "",
                "verifier_model": args.verifier_model or "random",
                "N": N,
                "selection_mode": args.selection_mode,
                "verifier_style": args.verifier_style,
                "verifier_scoring": args.verifier_scoring,
                "n_problems": verifier_res["n"],
                "verifier_pass@1": f"{verifier_res['pass@1']:.4f}",
                "oracle_pass@1": f"{baselines.get('oracle_pass@1', 'N/A')}",
                "random_pass@1": f"{baselines.get('random_pass@1', 'N/A')}",
                "split": args.split,
                "lcb_version": args.lcb_version,
                "test_set": args.test_set,
            }
        )
    log.info(f"Results → {results_path}  |  summary → {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Best-of-N evaluation on LiveCodeBench")

    # Models
    p.add_argument("--generator_model", type=str, default=None, help="Path/name of code generator LLM")
    p.add_argument("--verifier_model", type=str, default=None, help="Path/name of verifier LLM")

    # BoN config
    p.add_argument("--N", type=int, default=8, help="Candidates to generate per problem")
    p.add_argument("--selection_mode", choices=["list", "pairwise"], default="list", help="list: all-in-context single verdict; pairwise: C(N,2) comparisons")

    # Verifier config
    p.add_argument("--verifier_style", choices=["thinking", "instruct"], default="thinking", help="thinking: R1-style with <think> prefix; instruct: COT system prompt")
    p.add_argument("--verifier_scoring", action="store_true", help="Use 0-10 scoring prompts instead of letter-selection prompts")
    p.add_argument("--no_bias_correction", action="store_true", help="Skip running both orderings for pairwise position-bias correction")
    p.add_argument("--verifier_max_tokens", type=int, default=16384)

    # Generator config
    p.add_argument("--generator_max_tokens", type=int, default=2048)
    p.add_argument("--generator_temperature", type=float, default=0.8)

    # Dataset config
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--lcb_version", type=str, default="release_v6", help="LiveCodeBench version tag (e.g. release_v6)")
    p.add_argument("--max_problems", type=int, default=0, help="Limit to first K problems (0 = all)")
    p.add_argument("--test_set", choices=["private", "public"], default="private", help="Which test cases to use for execution")

    # Execution
    p.add_argument("--execution_timeout", type=float, default=10.0, help="Per-test-case execution timeout in seconds")
    p.add_argument("--compute_baselines", action="store_true", help="Also compute oracle and random selection baselines (requires executing all N candidates per problem)")

    # Caching / output
    p.add_argument("--output_dir", type=str, default="outputs/bon_livecodebench")
    p.add_argument("--candidates_cache", type=str, default=None, help="Path to candidates .pkl from a previous run (skips generation)")
    p.add_argument("--selections_cache", type=str, default=None, help="Path to selections .pkl from a previous run (skips verifier)")

    args = p.parse_args()

    if not args.generator_model and not args.candidates_cache:
        p.error("Provide --generator_model or --candidates_cache (to skip generation)")

    main(args)
