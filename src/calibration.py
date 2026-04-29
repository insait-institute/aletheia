import argparse
import csv
import logging
import math
import os
import pickle
import random
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from evaluate import _create_prompts
from transformers import AutoTokenizer
from utils import run_inference

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1

OPTIONS = ["A", "B", "C", "D", "E"]
BOX_PREFIX = "\\boxed{"
CACHE_CSV = Path(__file__).parent / "outputs" / "eval_results_list_16.csv"


def _option_token_ids(tokenizer) -> Dict[str, List[int]]:
    candidates = {}
    for opt in OPTIONS:
        single = tokenizer.encode(opt, add_special_tokens=False)
        leading = tokenizer.encode(f" {opt}", add_special_tokens=False)
        ids = set()
        if len(single) == 1:
            ids.add(single[0])
        if len(leading) == 1:
            ids.add(leading[0])
        candidates[opt] = list(ids)
    return candidates


def _dist_from_position_logprobs(pos_logprobs, option_ids: Dict[str, List[int]], num_candidates: int) -> Optional[Dict[str, float]]:
    if pos_logprobs is None:
        return None
    valid_options = OPTIONS[:num_candidates]
    raw = {}
    for opt in valid_options:
        best = -math.inf
        for tid in option_ids[opt]:
            if tid in pos_logprobs:
                lp = pos_logprobs[tid].logprob
                if lp > best:
                    best = lp
        raw[opt] = best
    if all(v == -math.inf for v in raw.values()):
        return None
    finite = [v for v in raw.values() if v != -math.inf]
    floor = min(finite) - 20.0 if finite else -50.0
    raw = {k: (v if v != -math.inf else floor) for k, v in raw.items()}
    m = max(raw.values())
    exps = {k: math.exp(v - m) for k, v in raw.items()}
    Z = sum(exps.values())
    return {k: v / Z for k, v in exps.items()}


def _find_box_token_index(text: str, token_ids: List[int], tokenizer) -> Optional[int]:
    char_target = text.find(BOX_PREFIX)
    if char_target == -1:
        return None
    char_target += len(BOX_PREFIX)
    cumulative = ""
    for i, tid in enumerate(token_ids):
        cumulative += tokenizer.decode([tid])
        if len(cumulative) > char_target:
            return i
    return None


def extract_option_distribution_full(
    text: str,
    token_ids: List[int],
    logprobs_per_token,
    tokenizer,
    option_ids: Dict[str, List[int]],
    num_candidates: int,
) -> Optional[Dict[str, float]]:
    if logprobs_per_token is None:
        return None
    box_idx = _find_box_token_index(text, token_ids, tokenizer)
    if box_idx is None or box_idx >= len(logprobs_per_token):
        return None
    return _dist_from_position_logprobs(logprobs_per_token[box_idx], option_ids, num_candidates)


def _truncate_at_boxed(text: str) -> Optional[str]:
    idx = text.find(BOX_PREFIX)
    if idx == -1:
        return None
    return text[: idx + len(BOX_PREFIX)]


def _pick_continuation(completions, rng: random.Random) -> str:
    valid = [_truncate_at_boxed(c) for c in completions]
    valid = [t for t in valid if t is not None]
    if valid:
        return rng.choice(valid)
    return str(completions[rng.randrange(len(completions))]) + BOX_PREFIX


def _build_continuation_messages(orig_prompt, truncated_continuation: str) -> List[Dict[str, str]]:
    msgs = [dict(m) for m in orig_prompt]
    if msgs and msgs[-1]["role"] == "assistant":
        msgs[-1] = {"role": "assistant", "content": msgs[-1]["content"] + truncated_continuation}
    else:
        msgs.append({"role": "assistant", "content": truncated_continuation})
    return msgs


def _lookup_cache(eval_llm: str, data: str) -> Optional[Path]:
    if not CACHE_CSV.exists():
        return None
    with open(CACHE_CSV) as f:
        for row in csv.DictReader(f):
            if row["eval_llm"] == eval_llm and row["data"] == data:
                pkl = CACHE_CSV.parent / row["results_pkl_file"]
                if pkl.exists():
                    return pkl
    return None


def _run_fast_path(args, tokenizer, option_ids: Dict[str, List[int]], cached_pkl: Path) -> Tuple[List[Dict], List[float], List[int]]:
    log.info(f"Cache hit: {cached_pkl.name}. Using fast path (max_tokens=1).")
    df = pd.read_pickle(cached_pkl)
    rng = random.Random(args.seed)

    prompts, meta = [], []
    for _, row in df.iterrows():
        completions = list(row["completion"])
        truncated = _pick_continuation(completions, rng)
        new_prompt = _build_continuation_messages(row["prompt"], truncated)
        prompts.append(new_prompt)
        meta.append(
            {
                "chosen_answer": row["chosen_answer"],
                "num_candidates": int(row["num_candidates"]),
            }
        )

    completions = run_inference(
        prompts,
        args.eval_llm,
        temperature=0.0,
        max_tokens=1,
        max_model_len=args.max_tokens + 4096,
        dp_size=1,
        top_p=1.0,
        n=1,
        gpu_memory_utilization=0.95,
        logprobs=args.top_logprobs,
        tokenizer=tokenizer,
    )

    return _collect_results(completions, meta, option_ids, position=0)


def _run_slow_path(args, tokenizer, option_ids: Dict[str, List[int]]) -> Tuple[List[Dict], List[float], List[int]]:
    log.info("Cache miss. Falling back to full generation.")
    data = load_dataset(f"Aletheia-Bench/Aletheia-{args.data}", split="test")
    data = data.map(_create_prompts, fn_kwargs={"model_name": args.eval_llm}, num_proc=NUM_WORKERS, desc="Creating prompts")

    completions = run_inference(
        list(data["prompt"]),
        args.eval_llm,
        temperature=0.6,
        max_tokens=args.max_tokens,
        max_model_len=args.max_tokens + 4096,
        dp_size=1,
        top_p=0.95,
        n=1,
        gpu_memory_utilization=0.95,
        logprobs=args.top_logprobs,
        tokenizer=tokenizer,
    )

    meta = [{"chosen_answer": ca, "num_candidates": nc} for ca, nc in zip(data["chosen_answer"], data["num_candidates"])]

    records, confidences, predicted_correct = [], [], []
    for req, m in zip(completions, meta):
        out = req.outputs[0]
        dist = extract_option_distribution_full(out.text, list(out.token_ids), out.logprobs, tokenizer, option_ids, m["num_candidates"])
        rec, conf, corr = _record_from_dist(dist, m)
        records.append(rec)
        if conf is not None:
            confidences.append(conf)
            predicted_correct.append(corr)
    return records, confidences, predicted_correct


def _collect_results(completions, meta, option_ids, position: int) -> Tuple[List[Dict], List[float], List[int]]:
    records, confidences, predicted_correct = [], [], []
    for req, m in zip(completions, meta):
        out = req.outputs[0]
        lps = out.logprobs
        if lps is None or len(lps) <= position:
            records.append({"valid": False})
            continue
        dist = _dist_from_position_logprobs(lps[position], option_ids, m["num_candidates"])
        rec, conf, corr = _record_from_dist(dist, m)
        records.append(rec)
        if conf is not None:
            confidences.append(conf)
            predicted_correct.append(corr)
    return records, confidences, predicted_correct


def _record_from_dist(dist: Optional[Dict[str, float]], meta: Dict) -> Tuple[Dict, Optional[float], Optional[int]]:
    if dist is None:
        return {"valid": False}, None, None
    pred_opt, conf = max(dist.items(), key=lambda kv: kv[1])
    is_correct = int(pred_opt == meta["chosen_answer"])
    return (
        {
            "valid": True,
            "dist": dist,
            "conf": conf,
            "predicted": pred_opt,
            "correct": meta["chosen_answer"],
            "is_correct": is_correct,
            "num_candidates": meta["num_candidates"],
        },
        conf,
        is_correct,
    )


def compute_ece(confidences: List[float], correct: List[int], n_bins: int = 10):
    confidences = np.asarray(confidences)
    correct = np.asarray(correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        m = mask.sum()
        if m == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "conf": None, "acc": None})
            continue
        bin_conf = float(confidences[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (m / n) * abs(bin_conf - bin_acc)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m), "conf": bin_conf, "acc": bin_acc})
    return float(ece), bins


def compute_brier(confidences: List[float], correct: List[int]) -> float:
    return float(np.mean((np.asarray(confidences) - np.asarray(correct)) ** 2))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.eval_llm)
    chat_template = Path(__file__).parents[0] / "dsqwen_chat_template_dpo.jinja"
    tokenizer.chat_template = chat_template.read_text()
    option_ids = _option_token_ids(tokenizer)

    cached = None if args.no_cache else _lookup_cache(args.eval_llm, args.data)
    if cached is not None:
        records, confidences, predicted_correct = _run_fast_path(args, tokenizer, option_ids, cached)
        path = "fast"
    else:
        records, confidences, predicted_correct = _run_slow_path(args, tokenizer, option_ids)
        path = "slow"

    n_valid = len(confidences)
    n_total = len(records)
    log.info(f"Parsed {n_valid}/{n_total} responses successfully (path={path}).")

    ece, bins = compute_ece(confidences, predicted_correct, n_bins=args.n_bins)
    brier = compute_brier(confidences, predicted_correct)
    accuracy = float(np.mean(predicted_correct)) if predicted_correct else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    log.info(f"Accuracy: {accuracy * 100:.2f}")
    log.info(f"Mean confidence: {avg_conf * 100:.2f}")
    log.info(f"ECE ({args.n_bins} bins): {ece:.4f}")
    log.info(f"Brier score: {brier:.4f}")
    for b in bins:
        if b["n"] == 0:
            continue
        log.info(f"  bin ({b['lo']:.2f}, {b['hi']:.2f}]: n={b['n']}, conf={b['conf']:.3f}, acc={b['acc']:.3f}")

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rid = str(uuid.uuid4())[:8]
    pkl = out_dir / f"calibration_{rid}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "path": path,
                "records": records,
                "ece": ece,
                "brier": brier,
                "accuracy": accuracy,
                "mean_confidence": avg_conf,
                "bins": bins,
            },
            f,
        )

    csv_path = out_dir / "calibration_results.csv"
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["eval_llm", "data", "path", "n_total", "n_valid", "accuracy", "mean_confidence", "ece", "brier", "results_pkl"],
        )
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "eval_llm": args.eval_llm,
                "data": args.data,
                "path": path,
                "n_total": n_total,
                "n_valid": n_valid,
                "accuracy": f"{accuracy * 100:.2f}",
                "mean_confidence": f"{avg_conf * 100:.2f}",
                "ece": f"{ece:.4f}",
                "brier": f"{brier:.4f}",
                "results_pkl": pkl.name,
            }
        )
    log.info(f"Saved calibration data to {pkl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_llm", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, choices=["CRB", "Heldout", "Strong", "Hard", "Adv"])
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cache", action="store_true", help="Skip cache and force full generation.")
    args = parser.parse_args()
    main(args)
