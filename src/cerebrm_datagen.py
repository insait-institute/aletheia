import os
import random
from itertools import combinations

import numpy as np
import polars as pl
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

NUM_WORKERS = len(os.sched_getaffinity(0))
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
WEAK_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-Coder-7B-Instruct",
    "gemma-2-9b-it",
    "deepseek-coder-6.7b-instruct",
]
STRONG_MODELS = [
    "Llama-3.1-70B-Instruct",
    "Qwen2.5-Coder-32B-Instruct",
    "gemma-2-27b-it",
    "deepseek-coder-33b-instruct",
]
MAX_COMBOS_PER_LEN = 3  # to limit dataset size
LIST_REWARD_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Enclose your final answer in the format \\boxed{{[[X]]}}, where [[X]] is your chosen option among the candidates. Do not provide any explanations or additional text. Your response should be exactly one of the options enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""  # noqa: E501

JUDGELRM_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and two candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to assign a score between 0 and 10 to each candidate, with 10 indicating a perfect solution that passes all test cases, 5 indicating a solution that would pass some test cases but not all, and 0 indicating a solution that fails all test cases. Output your final answer in the format \\boxed{{[<score_candidate_A>, <score_candidate_B>]}}. Do not provide any explanations or additional text. Your response should be a list of exactly two numbers between 0 and 10, enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""  # noqa: E501

DS_GRM_PROMPT = """
You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to assign a score between 0 and 10 to EACH candidate, with 10 indicating a perfect solution that passes all test cases, 5 indicating a solution that would pass some test cases but not all, and 0 indicating a solution that fails all test cases. Output your final answer in the format \\boxed{{[<score_candidate_A>, <score_candidate_B>, <score_candidate_C>, ...]}} depending on the number of candidates. Do not provide any explanations or additional text. Your response should be a list of numbers between 0 and 10, enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""  # noqa: E501


def _remove_md(code, lang):
    return code.split(f"```{lang}")[-1].split("```")[0].strip()


def _remove_all_md(example):
    example["candidates"] = [_remove_md(c, example["language"]) for c in example["candidates"]]
    return example


def downsample_equalize_diversify(
    hf_dataset,
    target_total: int = 50_000,
    per_group: int = 12_500,
    group_col: str = "num_candidates",
    id_col: str = "prompt_id",
    min_num_ids: int = 6,
    group_values: list = None,
    seed: int = 42,
    sample_with_replacement: bool = False,
):
    """
    Downsample a HuggingFace Dataset to `target_total` rows by:
      1) Equalizing groups defined by `group_col` to `per_group` rows each (mandatory).
      2) Within each group, prefer unique id_col (prompt_id) rows to maximize diversity.
         If there are fewer unique prompt_ids than per_group, fill the remainder with
         additional rows (preferably other rows from that group). Optionally allow
         sampling with replacement if there aren't enough rows.

    Parameters:
      hf_dataset: datasets.Dataset
      target_total: final target number of rows (default 50_000)
      per_group: rows per chosen group (default 12_500)
      group_col: which column to equalize on (default "num_candidates")
      id_col: which column to use for diversity (default "prompt_id")
      min_num_ids: minimum number of entries per id col. If less, these instances are used for the test set
      group_values: explicit list of group values to use (if None, top N by count are chosen)
      seed: RNG seed for reproducibility
      sample_with_replacement: allow sampling with replacement if a group lacks rows

    Returns:
      train_data: A new datasets.Dataset containing the sampled rows with any num_candidates.
      test_data: A new datasets.Dataset containing the dropped rows with num_candidates=2.
    """
    rng = np.random.RandomState(seed)

    # Basic checks
    if target_total % per_group != 0:
        raise ValueError("target_total must be an integer multiple of per_group (or change per_group).")
    num_groups_needed = target_total // per_group

    # Convert to pandas for flexible grouping and indexing (90k is small; if > millions use streaming)
    df = hf_dataset.to_pandas()
    # keep the dataset row indices so we can re-select from the original dataset
    df["_orig_index"] = df.index

    # Determine which group values to use
    group_counts = df[group_col].value_counts()
    if group_values is None:
        if len(group_counts) < num_groups_needed:
            raise ValueError(
                f"Found only {len(group_counts)} distinct {group_col} values but need {num_groups_needed} groups. Either reduce per_group/target_total or provide 'group_values' explicitly."
            )
        chosen_groups = group_counts.index[:num_groups_needed].tolist()
    else:
        chosen_groups = list(group_values)
        if len(chosen_groups) != num_groups_needed:
            raise ValueError("Length of group_values must equal target_total // per_group")

    selected_indices = []

    for g in chosen_groups:
        sub = df[df[group_col] == g].copy()
        if sub.shape[0] == 0:
            raise ValueError(f"No rows found for group {g}")

        # map prompt_id -> list of original row indices
        grouped = sub.groupby(id_col)["_orig_index"].apply(list)
        unique_ids = grouped.index.tolist()
        qualifying_ids = [pid for pid in unique_ids if len(grouped.loc[pid]) > min_num_ids]
        n_unique = len(qualifying_ids)

        # 1) If we have enough unique prompt_ids, choose per_group of them (max diversity)
        if n_unique >= per_group:
            chosen_ids = rng.choice(qualifying_ids, size=per_group, replace=False)
            for pid in chosen_ids:
                rows_for_pid = grouped.loc[pid]
                pick = rng.choice(rows_for_pid)  # choose one row per prompt_id
                selected_indices.append(int(pick))
        else:
            # take one row per unique prompt_id first
            for pid in qualifying_ids:
                rows_for_pid = grouped.loc[pid]
                pick = rng.choice(rows_for_pid)
                selected_indices.append(int(pick))

            remaining = per_group - n_unique
            # pool of remaining rows in this group (excluding already selected rows)
            already = set(selected_indices)
            pool = sub[(~sub["_orig_index"].isin(already)) & (sub[id_col].isin(qualifying_ids))]["_orig_index"].tolist()

            if len(pool) >= remaining:
                extra = rng.choice(pool, size=remaining, replace=False).tolist()
            else:
                if sample_with_replacement:
                    # allow re-sampling from entire group's rows
                    all_rows = sub["_orig_index"].tolist()
                    extra = rng.choice(all_rows, size=remaining, replace=True).tolist()
                else:
                    raise ValueError(
                        f"Group {g} does not have enough rows to reach {per_group} (need {remaining} more). Enable sample_with_replacement=True to allow duplicates, or choose different groups."
                    )
            selected_indices.extend([int(x) for x in extra])

    # Sanity check final size
    if len(selected_indices) != target_total:
        # defensive: if we somehow got duplication or mismatch, trim or pad
        if len(selected_indices) > target_total:
            selected_indices = rng.choice(selected_indices, size=target_total, replace=False).tolist()
        else:
            # small padding: sample from remaining rows not yet selected (prefer unique prompt_ids globally)
            remaining_pool = df[~df["_orig_index"].isin(selected_indices)]["_orig_index"].tolist()
            need = target_total - len(selected_indices)
            if len(remaining_pool) >= need:
                extra = rng.choice(remaining_pool, size=need, replace=False).tolist()
            else:
                if sample_with_replacement:
                    extra = rng.choice(df["_orig_index"].tolist(), size=need, replace=True).tolist()
                else:
                    raise ValueError("Unable to reach target_total with available rows.")
            selected_indices.extend([int(x) for x in extra])

    # final selection: keep order random
    rng.shuffle(selected_indices)
    # build new HuggingFace Dataset using dataset.select (indices refer to original dataset order)
    train_ds = hf_dataset.select(selected_indices)
    train_prompts = train_ds["prompt_id"]

    def _test_ds(example):
        return example["num_candidates"] == 2 and example["prompt_id"] not in train_prompts

    test_ds = hf_dataset.filter(_test_ds, num_proc=NUM_WORKERS)
    # (Optional) include metadata column describing selection group if you want; omitted here
    return train_ds, test_ds


def add_results_to_completions(example, dct):
    example["completions"] = dct[example["id"]][example["generator"]][0]
    example["description"] = dct[example["id"]][example["generator"]][1]
    return example


def _count_prompt_tokens(example):
    potential_answers = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i[2]}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i[2]}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    max_prompt_tokens = -1
    for reward_type in ["list_dist", "judge_lrm", "ds_grm"]:
        if reward_type == "list_em" or reward_type == "list_dist":
            prompt = [
                {
                    "role": "user",
                    "content": LIST_REWARD_PROMPT.format(
                        question=example["query"],
                        candidates=candidate_str,
                        valid_options=", ".join(potential_answers),
                    ).strip(),
                },
            ]
        elif reward_type == "judge_lrm":
            prompt = [
                {
                    "role": "user",
                    "content": JUDGELRM_PROMPT.format(
                        question=example["query"],
                        candidates=candidate_str,
                    ).strip(),
                },
            ]
        elif reward_type == "ds_grm":
            prompt = [
                {
                    "role": "user",
                    "content": DS_GRM_PROMPT.format(
                        question=example["query"],
                        candidates=candidate_str,
                    ).strip(),
                },
            ]
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenized_prompt = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
        max_prompt_tokens = max(max_prompt_tokens, len(tokenized_prompt))
    example["max_prompt_tokens"] = max_prompt_tokens
    return example


def _by_prompt_tokens(example, max_tokens=4096):
    return example["max_prompt_tokens"] <= max_tokens


def _add_chosen_rejected(example):
    example["chosen"] = example["candidates"][example["chosen_position"]]
    example["rejected"] = example["candidates"][~example["chosen_position"]]
    example["chosen_pass_rate"] = example["pass_rates"][example["chosen_position"]]
    example["rejected_pass_rate"] = example["pass_rates"][~example["chosen_position"]]
    return example


def pass_rates_and_filter(example):
    example["pass_rates"] = [x / (x + y) for (x, y) in zip(example["num_passed"], example["num_failed"])]
    tokenized_completions = tokenizer(example["completions"], padding=False, truncation=False)["input_ids"]

    example["completions"] = [x for x, z in zip(example["completions"], tokenized_completions) if len(z) <= 800]
    example["pass_rates"] = [y for y, z in zip(example["pass_rates"], tokenized_completions) if len(z) <= 800]
    example["num_surviving"] = len(example["completions"])
    return example


def main():
    data = load_dataset("wetsoledrysoul/CCPlus-Executed", "all")["train"]
    completions_data = load_dataset("wetsoledrysoul/ccplus_completions", "all")["train"]

    table1 = pl.from_arrow(data.data.table)
    table2 = pl.from_arrow(completions_data.data.table)
    joined = table1.join(table2, on=["id", "generator", "language"], how="inner")

    data = Dataset(joined.to_arrow())

    # 90% of the completions are less than 800 tokens. Set that as the limit
    data = data.map(pass_rates_and_filter, num_proc=NUM_WORKERS, remove_columns=["num_passed", "num_failed"])
    answer_list = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"]
    weak_easy, weak_hard, strong_easy = [], [], []
    for example in data:
        example["description"] = example["description"].split("\nExample\n")[0].split("\nExamples\n")[0].strip()

        pass_rates = example["pass_rates"]
        codes = example["completions"]
        if not codes:
            continue
        chosen = list(set([x for x, y in zip(codes, pass_rates) if y == 1.0]))

        if not len(chosen):
            continue

        # Cluster codes with same pass rates together to maintain list diversity
        ratio_to_codes = {}
        for code, ratio in zip(codes, pass_rates):
            if ratio == 1.0:
                continue
            if ratio not in ratio_to_codes:
                ratio_to_codes[ratio] = []
            ratio_to_codes[ratio].append(code)

        easy_ratios = [ratio for ratio in ratio_to_codes.keys() if ratio <= 0.5]
        hard_ratios = [ratio for ratio in ratio_to_codes.keys() if 0.7 <= ratio <= 0.9]

        maxlen_easy = min(len(easy_ratios), 5)
        maxlen_hard = min(len(hard_ratios), 5)

        if maxlen_easy >= 2:
            for ith_len in range(1, maxlen_easy):
                # create all possible combinations of easy ratios of length ith_len
                all_ratio_combos = [list(x) for x in combinations(easy_ratios, ith_len)]
                # sample a maximum of MAX_COMBOS_PER_LEN combinations of unique ratios per list length
                selected_combos = random.sample(all_ratio_combos, min(MAX_COMBOS_PER_LEN, len(all_ratio_combos)))
                # for each combination, select one code from each ratio bucket and combine with one chosen code
                for ratios in selected_combos:
                    codes_list = [random.sample(chosen, 1)[0]]
                    for ratio in ratios:
                        codes_list.append(random.sample(ratio_to_codes[ratio], 1)[0])
                    pr_list = [1.0] + [x for x in ratios]
                    paired = list(zip(codes_list, pr_list))
                    random.shuffle(paired)
                    codes_list, pr_list = zip(*paired)

                    # Add to appropriate list based on model type
                    if example["generator"] in WEAK_MODELS:
                        weak_easy.append(
                            {
                                "prompt_id": example["id"],
                                "query": example["description"],
                                "generator": example["generator"],
                                "language": example["language"],
                                "num_candidates": len(codes_list),
                                "candidates": codes_list,
                                "chosen_position": pr_list.index(1.0),
                                "chosen_answer": answer_list[pr_list.index(1.0)],
                                "pass_rates": pr_list,
                                "easy_hard": "easy",
                                "weak_strong": "weak",
                                "source": example["source"],
                            }
                        )
                    else:  # STRONG_MODELS
                        if len(codes_list) == 2:  # we only care about pairs for strong_easy
                            strong_easy.append(
                                {
                                    "prompt_id": example["id"],
                                    "query": example["description"],
                                    "generator": example["generator"],
                                    "language": example["language"],
                                    "num_candidates": len(codes_list),
                                    "candidates": codes_list,
                                    "chosen_position": pr_list.index(1.0),
                                    "chosen_answer": answer_list[pr_list.index(1.0)],
                                    "pass_rates": pr_list,
                                    "easy_hard": "easy",
                                    "weak_strong": "strong",
                                    "source": example["source"],
                                }
                            )
        if maxlen_hard >= 2:
            if example["generator"] in STRONG_MODELS:
                continue
            # we only care about pairs and weak_hard
            selected_ratios = random.sample(hard_ratios, min(MAX_COMBOS_PER_LEN, len(hard_ratios)))
            for ratio in selected_ratios:
                codes_list = [random.sample(chosen, 1)[0], random.sample(ratio_to_codes[ratio], 1)[0]]
                pr_list = [1.0, ratio]
                paired = list(zip(codes_list, pr_list))
                random.shuffle(paired)
                codes_list, pr_list = zip(*paired)

                # Add to appropriate list based on model type
                weak_hard.append(
                    {
                        "prompt_id": example["id"],
                        "query": example["description"],
                        "generator": example["generator"],
                        "language": example["language"],
                        "num_candidates": len(codes_list),
                        "candidates": codes_list,
                        "chosen_position": pr_list.index(1.0),
                        "chosen_answer": answer_list[pr_list.index(1.0)],
                        "pass_rates": pr_list,
                        "easy_hard": "hard",
                        "weak_strong": "weak",
                        "source": example["source"],
                    }
                )

    weak_easy = Dataset.from_list(weak_easy)
    weak_hard = Dataset.from_list(weak_hard)
    strong_easy = Dataset.from_list(strong_easy)
    weak_easy = weak_easy.map(_count_prompt_tokens, num_proc=NUM_WORKERS, desc="Counting prompt tokens")
    weak_hard = weak_hard.map(_count_prompt_tokens, num_proc=NUM_WORKERS, desc="Counting prompt tokens")
    strong_easy = strong_easy.map(_count_prompt_tokens, num_proc=NUM_WORKERS, desc="Counting prompt tokens")
    weak_easy = weak_easy.filter(_by_prompt_tokens, fn_kwargs={"max_tokens": 4096}, num_proc=NUM_WORKERS, desc="Filtering by prompt tokens")
    weak_hard = weak_hard.filter(_by_prompt_tokens, fn_kwargs={"max_tokens": 4096}, num_proc=NUM_WORKERS, desc="Filtering by prompt tokens")
    strong_easy = strong_easy.filter(_by_prompt_tokens, fn_kwargs={"max_tokens": 4096}, num_proc=NUM_WORKERS, desc="Filtering by prompt tokens")

    ds = DatasetDict(
        {
            "weak_easy": weak_easy.shuffle(seed=42),
            "weak_hard": weak_hard.shuffle(seed=42),
            "strong_easy": strong_easy.shuffle(seed=42),
        }
    )

    weak_easy_train, weak_easy_test = downsample_equalize_diversify(weak_easy, target_total=50_000, per_group=12_500)
    ds = DatasetDict(
        {
            "train": weak_easy_train.shuffle(seed=42),
            "test_e2h": weak_hard.shuffle(seed=42),
            "test_w2s": strong_easy.shuffle(seed=42),
            "test_weak_easy": weak_easy_test.shuffle(seed=42),
        }
    )
    ds = ds.map(_remove_all_md, num_proc=NUM_WORKERS, desc="Removing markdown formatting")
    ds.push_to_hub("CodeShield/CerebRM-Dataset", private=True, max_shard_size="5GB", commit_message="update dataset with re-execution")
    ds.push_to_hub("wetsoledrysoul/CerebRM-Dataset", private=True, max_shard_size="5GB", commit_message="update dataset with re-execution")

    weak_easy_test = DatasetDict({"test": weak_easy_test.shuffle(seed=42)})
    weak_easy_test = weak_easy_test.map(
        _add_chosen_rejected,
        num_proc=NUM_WORKERS,
        desc="Removing markdown formatting",
        remove_columns=[
            "chosen_position",
            "chosen_answer",
            "pass_rates",
            "candidates",
            "num_candidates",
            "easy_hard",
            "weak_strong",
        ],
    )
    weak_easy_test.push_to_hub("CodeShield/RQ4-Set", private=True, max_shard_size="5GB", commit_message="update dataset with re-execution")
    weak_easy_test.push_to_hub("wetsoledrysoul/RQ4-Set", private=True, max_shard_size="5GB", commit_message="update dataset with re-execution")


if __name__ == "__main__":
    main()
    # select a random sample from each ratio bucket
