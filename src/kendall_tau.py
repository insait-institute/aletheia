import argparse
import csv
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from ranking_consistency import _build_preference_map, _load_dataset_metadata

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def compute_kendall_for_list(prefs: Dict[Tuple[int, int], int], pass_rates: List[float]) -> Optional[Tuple[float, float]]:
    n = len(pass_rates)
    if n < 2:
        return None
    win_counts = [0] * n
    for winner, _ in prefs.keys():
        win_counts[winner] += 1

    # If all predicted scores are tied, tau is undefined.
    if len(set(win_counts)) == 1 or len(set(pass_rates)) == 1:
        return None

    tau, _ = kendalltau(win_counts, pass_rates, variant="b")
    if tau is None or np.isnan(tau):
        return None
    return float(tau), float(np.std(win_counts))


def compute_kendall_tau(
    by_list: Dict[int, Dict[Tuple[int, int], int]],
    list_pass_rates: Dict[int, List[float]],
) -> Tuple[float, List[dict]]:
    scores, per_list = [], []
    for lid, prefs in by_list.items():
        prs = list_pass_rates.get(lid)
        if prs is None:
            continue
        result = compute_kendall_for_list(prefs, prs)
        if result is None:
            tau = -1
        else:
            tau, _ = result
        scores.append(tau)
        per_list.append({"list_id": lid, "n": len(prs), "kendall_tau": tau})
    mean_tau = float(np.mean(scores)) if scores else 0.0
    return mean_tau, per_list


def _process_pkl(pkl_path: Path, data_name: str) -> Optional[dict]:
    if not pkl_path.exists():
        log.warning(f"Missing pkl: {pkl_path}")
        return None
    with open(pkl_path, "rb") as f:
        saved = pickle.load(f)
    records = saved["records"]
    saved_args = saved.get("args", {})
    max_lists = saved_args.get("max_lists", 0)

    _, _, list_pass_rates = _load_dataset_metadata(data_name, max_lists)
    prefs_by_list = _build_preference_map(records)
    mean_tau, per_list = compute_kendall_tau(prefs_by_list, list_pass_rates)
    return {
        "mean_kendall_tau": mean_tau,
        "n_lists_scored": len(per_list),
        "per_list": per_list,
    }


def main(args):
    out_dir = Path(__file__).parent / "outputs"
    src_csv = out_dir / "ranking_results.csv"
    dst_csv = out_dir / "kendall_tau_results.csv"

    if not src_csv.exists():
        raise FileNotFoundError(f"{src_csv} not found")

    df = pd.read_csv(src_csv)
    if args.eval_llm:
        df = df[df["eval_llm"] == args.eval_llm]
    if args.data:
        df = df[df["data"] == args.data]

    if df.empty:
        log.warning("No matching rows in ranking_results.csv")
        return

    existing_keys = set()
    if dst_csv.exists() and not args.overwrite:
        existing = pd.read_csv(dst_csv)
        existing_keys = set(zip(existing["eval_llm"], existing["data"]))

    fieldnames = [
        "eval_llm",
        "data",
        "n_lists",
        "n_pairs",
        "n_valid",
        "n_lists_scored",
        "mean_kendall_tau",
        "results_pkl",
    ]

    write_header = not dst_csv.exists() or args.overwrite
    mode = "w" if args.overwrite else "a"
    with open(dst_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for _, row in df.iterrows():
            key = (row["eval_llm"], row["data"])
            # if key in existing_keys:
            #     log.info(f"Skipping already-computed {key}")
            #     continue
            pkl_path = out_dir / row["results_pkl"]
            log.info(f"Processing {key} from {pkl_path.name}")
            result = _process_pkl(pkl_path, row["data"])
            if result is None:
                continue
            log.info(f"  mean Kendall tau = {result['mean_kendall_tau']:.4f} over {result['n_lists_scored']} lists")
            writer.writerow(
                {
                    "eval_llm": row["eval_llm"],
                    "data": row["data"],
                    "n_lists": row["n_lists"],
                    "n_pairs": row["n_pairs"],
                    "n_valid": row["n_valid"],
                    "n_lists_scored": result["n_lists_scored"],
                    "mean_kendall_tau": f"{result['mean_kendall_tau']:.4f}",
                    "results_pkl": row["results_pkl"],
                }
            )
            f.flush()

    log.info(f"Wrote results to {dst_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_llm", type=str, default=None, help="Filter to one model")
    parser.add_argument("--data", type=str, default=None, help="Filter to one dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination csv")
    args = parser.parse_args()
    main(args)
