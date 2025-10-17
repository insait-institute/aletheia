import argparse
import ast
import csv
import logging
import os
import pickle
import random
import re
import statistics
import uuid
from pathlib import Path
from typing import List

from datasets import load_dataset

from cerebrm_prompts import DS_GRM_PROMPT, JUDGELRM_PROMPT, LIST_REWARD_PROMPT, RAFT_PROMPT_NOTHINK, RAFT_PROMPT_THINK, STAR_PROMPT
from utils import run_inference

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def extract_boxed_contents_score10(text: str) -> List[int]:
    """
    Extracts all contents within \\boxed{...} from a given text string,
    after normalizing braces.
    """
    # Match \boxed{...} with non-greedy content
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.search(pattern, text)
    try:
        matches = ast.literal_eval(matches.group(1))
        assert isinstance(matches, list) and all(isinstance(x, int) for x in matches)
    except Exception:
        matches = []
    return matches


def extract_boxed_contents_list(text: str) -> List[int]:
    """
    Extracts all contents within \\boxed{...} from a given text string,
    after normalizing braces.
    """
    # Match \boxed{...} with non-greedy content
    pattern = r"\\boxed\{(.*?)\}"
    matches = re.search(pattern, text)
    try:
        matches = matches.group(1)
    except Exception:
        matches = None
    return matches


def _create_prompts(example, model_name):
    model_name = model_name.lower()
    example["correct_ans"] = "[[A]]"
    candidate_str = f"[CANDIDATE_A]\n```{example['language']}\n{example['chosen']}\n```\n[/CANDIDATE_A]\n\n[CANDIDATE_B]\n```{example['language']}\n{example['rejected']}\n```\n[/CANDIDATE_B]"
    if random.random() > 0.5:
        example["correct_ans"] = "[[B]]"
        candidate_str = f"[CANDIDATE_A]\n```{example['language']}\n{example['rejected']}\n```\n[/CANDIDATE_A]\n\n[CANDIDATE_B]\n```{example['language']}\n{example['chosen']}\n```\n[/CANDIDATE_B]"

    PROMPT = LIST_REWARD_PROMPT
    valid_options = "[[A]], [[B]]"

    if "star" in model_name:
        PROMPT = STAR_PROMPT
    elif "raft" in model_name:
        if "deepseek" in model_name:
            PROMPT = RAFT_PROMPT_THINK
        else:
            PROMPT = RAFT_PROMPT_NOTHINK
    elif "judge_lrm" in model_name:
        PROMPT = JUDGELRM_PROMPT
        valid_options = None
    elif "ds_grm" in model_name:
        PROMPT = DS_GRM_PROMPT
        valid_options = None
    if valid_options:
        example["prompt"] = [
            {
                "role": "user",
                "content": PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                    valid_options=valid_options,
                ).strip(),
            },
        ]
    else:
        example["prompt"] = [
            {
                "role": "user",
                "content": PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                ).strip(),
            },
        ]
    return example


def interpret_scores(scores: List[int]) -> str:
    if len(scores) != 2:
        return "[[Invalid]]"  # Indeterminate if we don't have exactly two scores
    if scores[0] > scores[1]:
        return "[[A]]"
    elif scores[1] > scores[0]:
        return "[[B]]"
    else:
        return "[[Invalid]]"  # Indeterminate if scores are equal


def main(args):
    if args.data == "rq1":
        data = load_dataset("wetsoledrysoul/RQ1-Set")
        data = data["filtered"]
    elif args.data == "rq2":
        data = load_dataset("wetsoledrysoul/RQ2-Set")
        data = data["full"]
    elif args.data == "rq3":
        raise NotImplementedError("RQ3 dataset not yet available")
    elif args.data == "rq4":
        data = load_dataset("wetsoledrysoul/RQ4-Set")
    else:
        data = load_dataset("wetsoledrysoul/RQ4-Set", split="original")
    if args.reward_type is None:
        if "list_dist" in args.eval_llm:
            args.reward_type = "list_dist"
        elif "list_em" in args.eval_llm:
            args.reward_type = "list_em"
        elif "judge_lrm" in args.eval_llm:
            args.reward_type = "judge_lrm"
        elif "ds_grm" in args.eval_llm:
            args.reward_type = "ds_grm"
    data = data.map(_create_prompts, fn_kwargs={"model_name": args.eval_llm}, num_proc=NUM_WORKERS, desc="Creating prompts")
    prompts = list(data["prompt"])
    completions = run_inference(
        prompts,
        args.eval_llm,
        temperature=0.6,
        max_tokens=args.max_tokens,
        tp_size=1,
        top_p=0.95,
        n=args.K,
        gpu_memory_utilization=0.95,
    )
    completions = [[nth_response.text for nth_response in responses.outputs] for responses in completions]

    if args.reward_type in ["judge_lrm", "ds_grm"]:
        scores = [[extract_boxed_contents_score10(y) for y in x] for x in completions]
        model_answers = [[interpret_scores(y) for y in x] for x in scores]
    else:
        model_answers = [[extract_boxed_contents_list(y) for y in x] for x in completions]

    accuracies = []
    for correct_ans, answers in zip(data["correct_ans"], model_answers):
        if len(answers) == 1:
            accuracies.append({"SC": 1 if answers[0] == correct_ans else 0, "BoN": None})
        else:
            sc_ans = statistics.mode(answers)
            bon_ans = 1 if correct_ans in answers else 0
            accuracies.append({"SC": 1 if sc_ans == correct_ans else 0, "BoN": bon_ans})

    log.info(f"{args.eval_llm} accuracy")
    for metric in ["SC", "BoN"]:
        metric_values = [x[metric] for x in accuracies if x[metric] is not None]
        if metric_values:
            log.info(f"{metric}: = {sum(metric_values) / len(metric_values):.4f}")
    # Create a CSV file with results

    # Calculate overall metrics
    sc_values = [x["SC"] for x in accuracies if x["SC"] is not None]
    bon_values = [x["BoN"] for x in accuracies if x["BoN"] is not None]

    sc_accuracy = sum(sc_values) / len(sc_values) if sc_values else 0
    bon_accuracy = sum(bon_values) / len(bon_values) if bon_values else 0
    # Create filename with timestamp
    random_id = str(uuid.uuid4())[:8]
    pkl_filename = Path(__file__).parent / f"outputs/completions_{random_id}.pkl"
    csv_filename = Path(__file__).parent / "outputs/eval_results.csv"

    # Save completions to pickle file
    with open(pkl_filename, "wb") as f:
        pickle.dump(completions, f)
    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["eval_llm", "reward_type", "K", "data", "max_tokens", "SC_accuracy", "BoN_accuracy", "completions_pkl_file"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {
                "eval_llm": args.eval_llm,
                "reward_type": args.reward_type if args.reward_type else "-",
                "K": args.K,
                "data": args.data,
                "max_tokens": args.max_tokens,
                "SC_accuracy": f"{sc_accuracy:.4f}",
                "BoN_accuracy": f"{bon_accuracy:.4f}" if bon_values else "N/A",
                "completions_pkl_file": pkl_filename.name,
            }
        )

    log.info(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_llm", type=str, required=True, help="LLM to use for evaluation")
    parser.add_argument("--reward_type", type=str, default=None, choices=["list_em", "list_dist", "judge_lrm", "ds_grm"], help="Type of reward model prompt to use")
    parser.add_argument("--K", type=int, default=1, help="Number of samples to generate for each prompt")
    parser.add_argument("--data", type=str, default="heldout", choices=["rq1", "rq2", "rq3", "rq4", "heldout"], help="Which dataset to use for evaluation")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    main(args)
