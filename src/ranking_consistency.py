import argparse
import csv
import itertools
import logging
import math
import os
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from evaluate import _create_prompts, extract_boxed_contents_list
from utils import run_inference

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _build_pair_examples(example, idx):
    n = example["num_candidates"]
    if n < 3:
        return {"pairs": []}
    pairs = []
    for i, j in itertools.combinations(range(n), 2):
        pairs.append(
            {
                "list_id": idx,
                "i": i,
                "j": j,
                "query": example["query"],
                "language": example["language"],
                "candidates": [example["candidates"][i], example["candidates"][j]],
                "num_candidates": 2,
                "chosen_position": example["chosen_position"],
            }
        )
    return {"pairs": pairs}


def _flatten_pairs(dataset) -> List[dict]:
    flat = []
    for row in dataset:
        flat.extend(row["pairs"])
    return flat


def _interpret_answer(raw: Optional[str], pair: dict) -> Optional[int]:
    if raw is None:
        return None
    raw = raw.strip().upper()
    if raw == "A":
        return pair["i"]
    if raw == "B":
        return pair["j"]
    return None


def _build_preference_map(records: List[dict]) -> Dict[int, Dict[Tuple[int, int], int]]:
    by_list: Dict[int, Dict[Tuple[int, int], int]] = {}
    for r in records:
        if r["winner"] is None:
            continue
        lid = r["list_id"]
        winner = r["winner"]
        loser = r["j"] if winner == r["i"] else r["i"]
        by_list.setdefault(lid, {})[(winner, loser)] = 1
    return by_list


def _is_transitive_triple(prefs: Dict[Tuple[int, int], int], triple: Tuple[int, int, int]) -> Optional[bool]:
    a, b, c = triple
    wins = {a: 0, b: 0, c: 0}
    seen = 0
    for x, y in itertools.combinations([a, b, c], 2):
        if (x, y) in prefs:
            wins[x] += 1
            seen += 1
        elif (y, x) in prefs:
            wins[y] += 1
            seen += 1
    if seen < 3:
        return None
    return sorted(wins.values()) == [0, 1, 2]


def compute_transitivity(by_list: Dict[int, Dict[Tuple[int, int], int]], list_sizes: Dict[int, int]):
    total, transitive = 0, 0
    per_list = []
    for lid, prefs in by_list.items():
        n = list_sizes.get(lid, 0)
        if n < 3:
            continue
        list_total, list_trans = 0, 0
        for triple in itertools.combinations(range(n), 3):
            verdict = _is_transitive_triple(prefs, triple)
            if verdict is None:
                continue
            list_total += 1
            list_trans += int(verdict)
        per_list.append({"list_id": lid, "n": n, "triples": list_total, "transitive": list_trans})
        total += list_total
        transitive += list_trans
    rate = transitive / total if total else 0.0
    return rate, total, transitive, per_list


def _dcg(relevances: List[float]) -> float:
    return sum((2.0**rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances))


def compute_ndcg_for_list(prefs: Dict[Tuple[int, int], int], pass_rates: List[float]) -> Optional[float]:
    n = len(pass_rates)
    win_counts = {i: 0 for i in range(n)}
    for winner, _ in prefs.keys():
        win_counts[winner] += 1

    predicted_order = sorted(range(n), key=lambda i: win_counts[i], reverse=True)
    predicted_rels = [pass_rates[i] for i in predicted_order]
    ideal_rels = sorted(pass_rates, reverse=True)

    idcg = _dcg(ideal_rels)
    if idcg == 0.0:
        return None
    return _dcg(predicted_rels) / idcg


def compute_ndcg(
    by_list: Dict[int, Dict[Tuple[int, int], int]],
    list_pass_rates: Dict[int, List[float]],
) -> Tuple[float, List[dict]]:
    scores, per_list = [], []
    for lid, prefs in by_list.items():
        prs = list_pass_rates.get(lid)
        if prs is None:
            continue
        ndcg = compute_ndcg_for_list(prefs, prs)
        if ndcg is None:
            continue
        scores.append(ndcg)
        per_list.append({"list_id": lid, "n": len(prs), "ndcg": ndcg})
    mean_ndcg = float(np.mean(scores)) if scores else 0.0
    return mean_ndcg, per_list


def compute_pairwise_accuracy(records: List[dict], list_pass_rates: Dict[int, List[float]]) -> Tuple[float, int, int]:
    correct, total = 0, 0
    for r in records:
        if r["winner"] is None:
            continue
        prs = list_pass_rates.get(r["list_id"])
        if prs is None:
            continue
        i, j = r["i"], r["j"]
        if prs[i] == prs[j]:
            continue
        gt_winner = i if prs[i] > prs[j] else j
        total += 1
        correct += int(r["winner"] == gt_winner)
    rate = correct / total if total else 0.0
    return rate, correct, total


def _load_dataset_metadata(data_name: str, max_lists: int):
    raw = load_dataset(f"Aletheia-Bench/Aletheia-{data_name}", split="test")
    raw = raw.filter(lambda x: x["num_candidates"] >= 3)
    if max_lists and len(raw) > max_lists:
        raw = raw.select(range(max_lists))
    list_sizes = {i: row["num_candidates"] for i, row in enumerate(raw)}
    list_pass_rates = {i: row["pass_rates"] for i, row in enumerate(raw)}
    return raw, list_sizes, list_pass_rates


def _run_random_baseline(raw) -> List[dict]:
    pair_dataset = raw.map(_build_pair_examples, with_indices=True, num_proc=NUM_WORKERS, desc="Building pairs")
    pairs = _flatten_pairs(pair_dataset)
    log.info(f"Generated {len(pairs)} pairwise comparisons for random baseline")
    rng = np.random.default_rng(0)
    picks = rng.integers(0, 2, size=len(pairs))
    records = []
    for pair, pick in zip(pairs, picks):
        pick_a = bool(pick == 0)
        winner = pair["i"] if pick_a else pair["j"]
        records.append(
            {
                "list_id": pair["list_id"],
                "i": pair["i"],
                "j": pair["j"],
                "raw": "A" if pick_a else "B",
                "winner": winner,
            }
        )
    return records


def _run_pairwise_inference(args, raw) -> List[dict]:
    pair_dataset = raw.map(_build_pair_examples, with_indices=True, num_proc=NUM_WORKERS, desc="Building pairs")
    pairs = _flatten_pairs(pair_dataset)
    log.info(f"Generated {len(pairs)} pairwise comparisons")

    pair_ds = Dataset.from_list(pairs)
    pair_ds = pair_ds.map(
        lambda ex: _create_prompts(ex, args.eval_llm),
        num_proc=NUM_WORKERS,
        desc="Creating prompts",
    )

    completions = run_inference(
        list(pair_ds["prompt"]),
        args.eval_llm,
        temperature=0.6,
        max_tokens=args.max_tokens,
        max_model_len=args.max_tokens + 4096,
        dp_size=1,
        top_p=0.95,
        n=1,
        gpu_memory_utilization=0.95,
    )

    records = []
    for pair, req in zip(pairs, completions):
        raw_ans = extract_boxed_contents_list(req.outputs[0].text)
        winner = _interpret_answer(raw_ans, pair)
        records.append(
            {
                "list_id": pair["list_id"],
                "i": pair["i"],
                "j": pair["j"],
                "raw": raw_ans,
                "winner": winner,
            }
        )
    return records


def main(args):
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ranking_results.csv"
    rid = str(uuid.uuid4())[:8]
    pkl = out_dir / f"ranking_{rid}.pkl"

    # if csv_path.exists():
    #     existing = pd.read_csv(csv_path)
    #     if ((existing["eval_llm"] == args.eval_llm) & (existing["data"] == args.data)).any():
    #         log.info(f"Already computed: eval_llm={args.eval_llm}, data={args.data} — skipping.")
    #         return

    if args.from_pkl:
        csv_path = out_dir / "ranking_results.csv"
        existing_results = pd.read_csv(csv_path)
        relevant_row = existing_results[(existing_results["eval_llm"] == args.eval_llm) & (existing_results["data"] == args.data)].reset_index(drop=True)
        if len(relevant_row) < 1:
            raise ValueError(f"No relevant row found in existing_results for model {args.eval_llm} and data {args.data}")
        pkl_path = out_dir / relevant_row["results_pkl"][0]
        log.info(f"Loading records from {pkl_path}")
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
        records = saved["records"]
        saved_args = saved.get("args", {})
        max_lists = args.max_lists or saved_args.get("max_lists", 0)
        raw, list_sizes, list_pass_rates = _load_dataset_metadata(args.data, max_lists)
        log.info(f"Reloaded {len(raw)} lists; recomputing metrics over {len(records)} stored pairwise records.")
    else:
        raw, list_sizes, list_pass_rates = _load_dataset_metadata(args.data, args.max_lists)
        log.info(f"Loaded {len(raw)} lists with ≥3 candidates from Aletheia-{args.data}")
        if args.eval_llm == "random":
            records = _run_random_baseline(raw)
        else:
            records = _run_pairwise_inference(args, raw)

    valid = sum(1 for r in records if r["winner"] is not None)
    log.info(f"Parsed {valid}/{len(records)} pairwise verdicts")

    prefs_by_list = _build_preference_map(records)

    trans_rate, total_triples, trans_triples, trans_per_list = compute_transitivity(prefs_by_list, list_sizes)
    mean_ndcg, ndcg_per_list = compute_ndcg(prefs_by_list, list_pass_rates)
    pair_acc, pair_correct, pair_total = compute_pairwise_accuracy(records, list_pass_rates)

    log.info(f"Transitivity rate: {trans_rate:.4f} ({trans_triples}/{total_triples} triples)")
    log.info(f"Mean NDCG: {mean_ndcg:.4f} (over {len(ndcg_per_list)} lists)")
    log.info(f"Pairwise accuracy: {pair_acc:.4f} ({pair_correct}/{pair_total} pairs)")

    with open(pkl, "wb") as f:
        pickle.dump(
            {
                "args": vars(args),
                "records": records,
                "trans_per_list": trans_per_list,
                "ndcg_per_list": ndcg_per_list,
                "trans_rate": trans_rate,
                "total_triples": total_triples,
                "trans_triples": trans_triples,
                "mean_ndcg": mean_ndcg,
                "pair_acc": pair_acc,
                "pair_correct": pair_correct,
                "pair_total": pair_total,
            },
            f,
        )

    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        fieldnames = [
            "eval_llm",
            "data",
            "n_lists",
            "n_pairs",
            "n_valid",
            "n_triples",
            "transitive",
            "trans_rate",
            "mean_ndcg",
            "results_pkl",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "eval_llm": args.eval_llm,
                "data": args.data,
                "n_lists": len(raw),
                "n_pairs": len(records),
                "n_valid": valid,
                "n_triples": total_triples,
                "transitive": trans_triples,
                "trans_rate": f"{trans_rate:.4f}",
                "mean_ndcg": f"{mean_ndcg:.4f}",
                "results_pkl": pkl.name,
            }
        )
    log.info(f"Saved ranking metrics to {pkl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_llm", type=str, required=True)
    parser.add_argument("--data", type=str, required=True, choices=["CRB", "Heldout", "Strong", "Hard", "Adv"])
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_lists", type=int, default=0, help="Limit number of lists. 0 = all.")
    parser.add_argument("--from_pkl", action="store_true", help="Whether to load results from existing pkl files")
    args = parser.parse_args()
    main(args)
