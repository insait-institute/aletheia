import os
import pickle
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

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
    example['chosen']=trace
    return example


def main():
    original = load_dataset("CodeShield/CerebRM-Dataset")["train"]
    original = original.map(_create_prompts, num_proc=NUM_WORKERS, desc="Creating prompts")
    original = original.to_pandas()[["idx", "query", "num_candidates", "chosen_answer", "source", "language", "generator", "prompt_id", "prompt"]]
    original.rename(columns={"generator": "code_generator"}, inplace=True)
    original.set_index("idx", inplace=True)
    full_df = []
    for model in ["1.5B", "7B", "14B", "32B", "R1"]:
        try:
            with open(Path(os.getenv("WORK")) / "think_dpo" / model / "completions.pkl", "rb") as f:
                completions = pickle.load(f)
            df = original.copy()
            df["completions"] = completions
            df["reward"] = df.apply(lambda x: [list_reward(resp, x["chosen_answer"]) for resp in x["completions"]], axis=1)
            df["model"] = model
            full_df.append(df)
        except FileNotFoundError:
            print(f"No completions found for model {model}")
            pass
        try:
            with open(Path(os.getenv("WORK")) / "think_dpo" / model / "regenerated" / "completions.pkl", "rb") as f:
                extra_completions = pickle.load(f)
            df = original.copy()
            df = df[df.index.isin(extra_completions.keys())]
            df["completions"] = df.index.map(extra_completions)
            df["reward"] = df.apply(lambda x: [list_reward(resp, x["chosen_answer"]) for resp in x["completions"]], axis=1)
            df["model"] = model
            full_df.append(df)
        except FileNotFoundError:
            print(f"No regenerated completions found for model {model}")
            pass

    full_df = pd.concat(full_df).reset_index(names="idx")
    full_df = full_df.explode(["completions", "reward"]).reset_index(drop=True)
    # compute the indices where all models failed
    allwrong_indices = full_df.groupby("idx")["reward"].max() != 1
    allcorrect_indices = full_df.groupby("idx")["reward"].min() != 0
    redo_indices = allwrong_indices[allwrong_indices].index.tolist() + allcorrect_indices[allcorrect_indices].index.tolist()
    print(f"Total examples where all models failed: {len(redo_indices)}. Regenerate data for these indices")
    if len(redo_indices) > 0:
        with open(Path(os.getenv("WORK")) / "think_dpo" / "allwrong_indices.txt", "w") as f:
            f.write("\n".join(map(str, redo_indices)))
        # full_df = full_df[~full_df["idx"].isin(redo_indices)].reset_index(drop=True)

    correct_df = full_df[full_df["reward"] == 1.0].reset_index(drop=True)
    incorrect_df = full_df[full_df["reward"] == 0.0].reset_index(drop=True)
    indices = full_df["idx"].unique().tolist()

    tgt_indices_by_model = {k: len(indices) // 3 for k in ["1.5B", "7B", "14B"]}

    final_data = []
    for idx in tqdm(indices, total=len(indices)):
        correct_candidates = correct_df[correct_df["idx"] == idx]
        if len(correct_candidates):
            correct = correct_candidates.sample(n=1, random_state=42)
            correct_completion = correct["completions"].values[0]
            correct_model = correct["model"].values[0]
        else:
            correct = incorrect_df[(incorrect_df["idx"] == idx) & (incorrect_df["model"] == "R1")].sample(n=1, random_state=42)
            true_answer = original.loc[idx, "chosen_answer"]
            correct_completion = add_answer_to_reasoning(correct["completions"].values[0], true_answer)
            correct_model = "R1"

        incorrect_model_ordering = [x[0] for x in sorted(tgt_indices_by_model.items(), key=lambda x: x[1], reverse=True)]
        for incorrect_model in incorrect_model_ordering:
            candidate_completions = incorrect_df[(incorrect_df["idx"] == idx) & (incorrect_df["model"] == incorrect_model)]["completions"]
            if len(candidate_completions):
                incorrect_completion = candidate_completions.sample(n=1, random_state=42)
                tgt_indices_by_model[incorrect_model] -= 1
                break

        final_data.append(
            {
                "idx": idx,
                "prompt": original.loc[idx, "prompt"],
                "chosen": correct_completion,
                "rejected": incorrect_completion.values[0],
                "chosen_model": correct_model,
                "rejected_model": incorrect_model,
                "prompt_id": original.loc[idx, "prompt_id"],
                "query": original.loc[idx, "query"],
                "num_candidates": original.loc[idx, "num_candidates"],
                "chosen_answer": original.loc[idx, "chosen_answer"],
                "language": original.loc[idx, "language"],
            }
        )
    final_data = pd.DataFrame(final_data)

    # cleanup chosen format for all instances
    final_data = Dataset.from_pandas(final_data)
    final_data = final_data.map(add_answer_to_reasoning_batch, num_proc=NUM_WORKERS, desc="Adding answers to reasoning")
    final_data.push_to_hub("wetsoledrysoul/CerebRM-Dataset-DPO", private=True, max_shard_size="5GB")
    final_data.push_to_hub("CodeShield/CerebRM-Dataset-DPO", private=True, max_shard_size="5GB")


if __name__ == "__main__":
    main()
