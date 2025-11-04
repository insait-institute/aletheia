# This scripts generates data for ThinkDPO experiments
# Begin with the CerebRM dataset, with 50,000 instances containing a list of candidates (from 2 to 5), of which exactly one is correct
# Prompt a suite of LLMs to select the correct candidate, and score these responses based on a binary correctness metric
# Make pairs of correct and incorrect responses to form preference data for DPO training
# Push this data to HuggingFace
import argparse
import os
import pickle
from pathlib import Path

import torch
from datasets import load_dataset

from cerebrm_prompts import LIST_REWARD_PROMPT
from utils import run_inference

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_prompts(example):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    example["prompt"] = [
        {
            "role": "user",
            "content": LIST_REWARD_PROMPT.format(
                question=example["query"],
                candidates=candidate_str,
                valid_options=", ".join(potential_answers),
            ).strip(),
        },
        {"role": "assistant", "content": "<think>\n"},  # to avoid bug in reward calculation in older trl versions
    ]
    return example


def main(args):
    data = load_dataset("CodeShield/CerebRM-Dataset")["train"]
    data = data.map(_create_prompts, num_proc=NUM_WORKERS, desc="Creating prompts")
    if args.wrong_indices_file:
        wrong_indices = Path(args.wrong_indices_file).read_text().splitlines()
        data = data.filter(lambda x: x["idx"] in wrong_indices, num_proc=NUM_WORKERS, desc="Filtering wrong answers")
    prompts = list(data["prompt"])
    # Generate completions using Deepseek-R1-Distill-Qwen
    model = f"deepseek-ai/Deepseek-R1-Distill-Qwen-{args.size}"
    if args.size == "R1":
        model = Path(os.getenv("WORK")) / "DS-R1"
    completions = run_inference(
        prompts,
        model,
        temperature=0.6,
        quantization="fp8" if args.size == "R1" else None,
        n=args.K,
        gpu_memory_utilization=0.95,
        tp_size=torch.cuda.device_count(),
        dp_size=1,
        max_tokens=16384,
        max_model_len=20480,
    )
    completions = [[nth_response.text for nth_response in responses.outputs] for responses in completions]
    save_dict = {i: c for i, c in zip(data["idx"], completions)}
    # store the results
    output_dir = Path(os.getenv("WORK")) / "think_dpo" / args.size
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wrong_indices_file:
        output_dir = output_dir / "regenerated"
        output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "completions.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Data generation complete for {args.size}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    (parser.add_argument("--size", type=str, choices=["1.5B", "7B", "14B", "32B", "R1"], required=True, help="Model size to use for data generation"),)
    parser.add_argument("--wrong-indices-file", type=str, default=None, help="Path to a file containing indices of wrong answers.  If not given, inference is done on all examples.")
    parser.add_argument("--K", type=int, default=4, help="Number of completions to generate per prompt.")
    args = parser.parse_args()
    main(args)
