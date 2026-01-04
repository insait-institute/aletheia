import os
import pickle

import pandas as pd
from datasets import Dataset, load_dataset

from cerebrm_prompts import LIST_REWARD_PROMPT
from cerebrm_rewards import extract_boxed_contents_list

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def list_reward(completion, chosen_answer, **kwargs):
    contents = completion.split("</think>")[-1].strip()
    model_answer = extract_boxed_contents_list(contents)
    return 1.0 if model_answer == chosen_answer else 0.0


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


def add_answer_to_reasoning(trace, answer_to_add):
    trace = trace.split("</think>")[0] + "</think>\n"
    trace += f"\\boxed{{{answer_to_add}}}"
    return trace


def add_answer_to_reasoning_batch(example):
    trace = example["chosen"].split("</think>")[0] + "</think>\n"
    trace += f"\\boxed{{{example['chosen_answer']}}}"
    example["chosen"] = trace
    return example


def add_think_tags(example):
    if not example["completion"].startswith("<think>"):
        example["completion"] = "<think>\n" + example["completion"].strip()
    example["prompt"] = example["prompt"][0]["content"]
    return example


def main():
    original = load_dataset("CodeShield/CerebRM-Dataset")["train"]
    original = original.map(_create_prompts, num_proc=NUM_WORKERS, desc="Creating prompts")
    original = original.to_pandas()[["idx", "query", "num_candidates", "chosen_answer", "source", "language", "generator", "prompt_id", "prompt"]]
    original.set_index("idx", inplace=True)

    with open("/scratch/vatsal_venkatkrishna/completions_0.pkl", "rb") as f:
        completions_0 = pickle.load(f)
    with open("/scratch/vatsal_venkatkrishna/completions_1.pkl", "rb") as f:
        completions_1 = pickle.load(f)
    completions = {**completions_0, **completions_1}
    full_df = original.copy()
    full_df["completion"] = completions
    full_df = full_df.reset_index(names="idx")
    full_df["reward"] = full_df.apply(lambda x: [list_reward(resp, x["chosen_answer"]) for resp in x["completion"]], axis=1)

    full_df = full_df.explode(["completion", "reward"]).reset_index(drop=True)
    # compute the indices where all models failed

    correct_df = full_df[full_df["reward"] == 1.0].reset_index(drop=True)

    tgt_size = len(original)
    one_per_idx = correct_df.sample(frac=1, random_state=42).drop_duplicates(subset="idx", keep="first").reset_index(names="full_dset_idx")
    print(f"Sampled {len(one_per_idx)} instances initially, with one entry per original training index")
    remaining_data = correct_df[~correct_df.index.isin(one_per_idx.full_dset_idx)]
    # sample either how much is needed to get to the original length, or how much is available
    num_to_sample = min(tgt_size - len(one_per_idx), len(remaining_data))
    additional_data = remaining_data.sample(n=num_to_sample, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(additional_data)} additional instances randomly")
    final_ds = pd.concat([one_per_idx.drop(["full_dset_idx"], axis=1), additional_data], axis=0).drop(["reward"], axis=1).reset_index(drop=True)
    print(f"Final dataset: {len(final_ds)} instances")

    final_data = Dataset.from_pandas(final_ds)
    breakpoint()
    final_data = final_data.map(add_think_tags, num_proc=NUM_WORKERS, desc="Adding <think> to reasoning traces")
    final_data.push_to_hub("wetsoledrysoul/CerebRM-Dataset-GenRM", private=True, max_shard_size="5GB")
    final_data.push_to_hub("CodeShield/CerebRM-Dataset-GenRM", private=True, max_shard_size="5GB")


if __name__ == "__main__":
    main()
