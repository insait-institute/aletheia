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

from cerebrm_prompts import LIST_REWARD_PROMPT, LIST_REWARD_PROMPT_COT
from utils import run_inference

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _shard(prompts: list, n: int) -> list[list]:
    k, r = divmod(len(prompts), n)
    return [prompts[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def _create_prompts(example, thinking=True):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    if thinking:
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
    else:
        example["prompt"] = [
            {"role": "system", "content": LIST_REWARD_PROMPT_COT.format(valid_options=", ".join(potential_answers))},
            {
                "role": "user",
                "content": f"Here is the coding question followed by the candidate solutions:\n[QUESTION]\n{example['query']}\n[/QUESTION]\n\n{candidate_str}\n\nYour response should be exactly in the specified format, without any extra characters or spaces. Anything else will be considered invalid.",
            },
        ]
    return example


def main(args):
    model_short_name = args.model.split("/")[-1].lower()
    data = load_dataset("CodeShield/CerebRM-Dataset")["train"]
    data = data.map(_create_prompts, fn_kwargs={"thinking": args.thinking}, num_proc=NUM_WORKERS, desc="Creating prompts")
    if args.wrong_indices_file:
        wrong_indices = Path(args.wrong_indices_file).read_text().splitlines()
        data = data.filter(lambda x: x["idx"] in wrong_indices, num_proc=NUM_WORKERS, desc="Filtering wrong answers")

    prompts = _shard(data, args.num_shards)[args.shard_num]
    prompts = list(data["prompt"])
    print(f"Generating data for {len(prompts)} examples using {args.model} model...")
    print(f"Example prompt: {prompts[0]}")
    # Generate completions using Deepseek-R1-Distill-Qwen
    completions = run_inference(
        prompts,
        args.model,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0,
        enable_expert_parallel=args.expert_parallel,
        n=args.K,
        gpu_memory_utilization=0.95,
        tp_size=torch.cuda.device_count(),
        dp_size=1,
        max_tokens=args.max_tokens,
        max_model_len=args.max_tokens + 4096,
        max_num_seqs=128,
        max_num_batched_tokens=25_000,
    )
    completions = [[nth_response.text for nth_response in responses.outputs] for responses in completions]
    save_dict = {i: c for i, c in zip(data["idx"], completions)}
    # store the results
    output_dir = Path(os.getenv("WORK")) / "think_dpo" / model_short_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.wrong_indices_file:
        output_dir = output_dir / "regenerated"
        output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"completions_{args.shard_num}.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print(f"Data generation complete for {args.model}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    (parser.add_argument("--model", type=str, help="Model to use for data generation"),)
    parser.add_argument("--wrong-indices-file", type=str, default=None, help="Path to a file containing indices of wrong answers.  If not given, inference is done on all examples.")
    parser.add_argument("--K", type=int, default=4, help="Number of completions to generate per prompt.")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the data into for generation.")
    parser.add_argument("--shard_num", type=int, default=0, help="Shard number to use for generation.")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum number of tokens to generate.")
    parser.add_argument("--expert_parallel", action="store_true", help="Whether to use expert parallelism during inference.")
    parser.add_argument("--thinking", action="store_true", help="Whether to use the thinking prompt format.")
    args = parser.parse_args()
    main(args)
