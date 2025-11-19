import os
import random
from itertools import combinations, product

import polars as pl
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from cerebrm_prompts import DS_GRM_PROMPT, LIST_REWARD_PROMPT

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
# to limit dataset size
MAX_PRODS_PER_LEN_WEAKEASY = 3
MAX_PRODS_PER_LEN_STRONGEASY = 1
MAX_PRODS_PER_LEN_WEAKHARD = 10
MAX_COMBOS_WEAKEASY = 5
MAX_COMBOS_STRONGEASY = 3
MAX_TEST_SIZE = 3000


def _remove_md(code, lang):
    return code.split(f"```{lang}")[-1].split("```")[0].strip()


def _remove_all_md(example):
    example["candidates"] = [_remove_md(c, example["language"]) for c in example["candidates"]]
    return example


def add_results_to_completions(example, dct):
    example["completions"] = dct[example["id"]][example["generator"]][0]
    example["description"] = dct[example["id"]][example["generator"]][1]
    return example


def _count_prompt_tokens(example):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    max_prompt_tokens = -1
    for reward_type in ["list_dist", "ds_grm"]:
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


def _by_prompt_id(example, prompt_ids):
    return example["id"] not in prompt_ids


def _add_chosen_rejected(example):
    example["chosen"] = example["candidates"][example["chosen_position"]]
    example["rejected"] = example["candidates"][~example["chosen_position"]]
    example["chosen_pass_rate"] = example["pass_rates"][example["chosen_position"]]
    example["rejected_pass_rate"] = example["pass_rates"][~example["chosen_position"]]
    return example


def pass_rates_and_filter(example):
    example["pass_rates"] = [x / (x + y) for (x, y) in zip(example["num_passed"], example["num_failed"])]
    # tokenized_completions = tokenizer(example["completions"], padding=False, truncation=False)["input_ids"]

    # example["completions"] = [x for x, z in zip(example["completions"], tokenized_completions) if len(z) <= 800]
    # example["pass_rates"] = [y for y, z in zip(example["pass_rates"], tokenized_completions) if len(z) <= 800]
    # example["num_surviving"] = len(example["completions"])
    return example


def main():
    data = load_dataset("wetsoledrysoul/CCPlus-Executed", "all")["train"]
    completions_data = load_dataset("wetsoledrysoul/ccplus_completions", "all")["train"]
    cerebrm_dataset = load_dataset("CodeShield/CerebRM-Dataset")["train"]
    train_pids = list(cerebrm_dataset["prompt_id"])

    table1 = pl.from_arrow(data.data.table)
    table2 = pl.from_arrow(completions_data.data.table)
    joined = table1.join(table2, on=["id", "generator", "language"], how="inner")

    data = Dataset(joined.to_arrow())
    data = data.filter(_by_prompt_id, fn_kwargs={"prompt_ids": train_pids}, num_proc=NUM_WORKERS, desc="Filtering out training prompts")
    # 90% of the completions are less than 800 tokens. Set that as the limit
    data = data.map(pass_rates_and_filter, num_proc=NUM_WORKERS, remove_columns=["num_passed", "num_failed"])
    answer_list = ["A", "B", "C", "D", "E"]
    weak_easy, weak_hard, strong_easy = [], [], []
    for example in tqdm(data, desc="Creating listwise eval sets", total=len(data)):
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
            if example["generator"] not in WEAK_MODELS + STRONG_MODELS:
                continue
            for ith_len in range(1, maxlen_easy):
                # create all possible combinations of easy ratios of length ith_len
                all_ratio_combos = [list(x) for x in combinations(easy_ratios, ith_len)]

                MAX_COMBOS = MAX_COMBOS_WEAKEASY if example["generator"] in WEAK_MODELS else MAX_COMBOS_STRONGEASY
                selected_combos = random.sample(all_ratio_combos, min(MAX_COMBOS, len(all_ratio_combos)))
                # for each combination, select one code from each ratio bucket and combine with one chosen code
                for ratios in selected_combos:
                    code_buckets = [chosen] + [ratio_to_codes[ratio] for ratio in ratios]
                    base_ratios = [1.0] + [x for x in ratios]
                    all_code_combos = list(product(*code_buckets))
                    MAX_PRODS = MAX_PRODS_PER_LEN_WEAKEASY if example["generator"] in WEAK_MODELS else MAX_PRODS_PER_LEN_STRONGEASY
                    all_code_combos = random.sample(all_code_combos, min(len(all_code_combos), MAX_PRODS))
                    for code_combo in all_code_combos:
                        paired = list(zip(code_combo, base_ratios))
                        random.shuffle(paired)
                        codes_list, pr_list = zip(*paired)
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
                        elif example["generator"] in STRONG_MODELS:
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
            if example["generator"] in STRONG_MODELS or example["generator"] not in WEAK_MODELS:
                continue
            for ith_len in range(1, maxlen_hard):
                # create all possible combinations of easy ratios of length ith_len
                all_ratio_combos = [list(x) for x in combinations(hard_ratios, ith_len)]
                # all_ratio_combos = random.sample(all_ratio_combos, min(len(all_ratio_combos), MAX_PRODS_PER_LEN_HARD))
                # for each combination, select one code from each ratio bucket and combine with one chosen code
                for ratios in all_ratio_combos:
                    code_buckets = [chosen] + [ratio_to_codes[ratio] for ratio in ratios]
                    base_ratios = [1.0] + [x for x in ratios]
                    all_code_combos = list(product(*code_buckets))
                    all_code_combos = random.sample(all_code_combos, min(len(all_code_combos), MAX_PRODS_PER_LEN_WEAKHARD))
                    for code_combo in all_code_combos:
                        paired = list(zip(code_combo, base_ratios))
                        random.shuffle(paired)
                        codes_list, pr_list = zip(*paired)
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
    print(f"Counts: len(weak_easy): {len(weak_easy)} len(weak_hard): {len(weak_hard)} len(strong_easy): {len(strong_easy)}")
    weak_easy = Dataset.from_list(weak_easy)
    weak_hard = Dataset.from_list(weak_hard)
    strong_easy = Dataset.from_list(strong_easy)

    weak_easy = weak_easy.map(_count_prompt_tokens, num_proc=NUM_WORKERS, desc="Counting prompt tokens")
    weak_hard = weak_hard.map(_count_prompt_tokens, num_proc=NUM_WORKERS, desc="Counting prompt tokens")
    strong_easy = strong_easy.map(_count_prompt_tokens, num_proc=NUM_WORKERS, desc="Counting prompt tokens")
    weak_easy = weak_easy.filter(_by_prompt_tokens, fn_kwargs={"max_tokens": 4096}, num_proc=NUM_WORKERS, desc="Filtering by prompt tokens")
    weak_hard = weak_hard.filter(_by_prompt_tokens, fn_kwargs={"max_tokens": 4096}, num_proc=NUM_WORKERS, desc="Filtering by prompt tokens")
    strong_easy = strong_easy.filter(_by_prompt_tokens, fn_kwargs={"max_tokens": 4096}, num_proc=NUM_WORKERS, desc="Filtering by prompt tokens")

    weak_easy = weak_easy.remove_columns(["max_prompt_tokens"])
    weak_hard = weak_hard.remove_columns(["max_prompt_tokens"])
    strong_easy = strong_easy.remove_columns(["max_prompt_tokens"])

    weak_easy = weak_easy.add_column("idx", [f"heldout_{x}" for x in range(len(weak_easy))])
    weak_hard = weak_hard.add_column("idx", [f"e2h_{x}" for x in range(len(weak_hard))])
    strong_easy = strong_easy.add_column("idx", [f"w2s_{x}" for x in range(len(strong_easy))])

    # create a new column to stratify by
    def _stratify_column(example):
        example["strat_col"] = f"{example['language']}_{example['num_candidates']}"
        return example

    def subsample_equal(ds, n=250):
        ds = ds.to_pandas()
        ds = ds.groupby(["num_candidates", "language"]).apply(lambda x: x.sample(n=min(len(x), n), random_state=42)).reset_index(drop=True)
        return Dataset.from_pandas(ds)

    # weak_easy_test = weak_easy.map(_stratify_column, num_proc=NUM_WORKERS, desc="Adding strat column")
    # weak_hard_test = weak_hard.map(_stratify_column, num_proc=NUM_WORKERS, desc="Adding strat column")
    # strong_easy_test = strong_easy.map(_stratify_column, num_proc=NUM_WORKERS, desc="Adding strat column")

    weak_easy_test = subsample_equal(weak_easy)
    weak_hard_test = subsample_equal(weak_hard)
    strong_easy_test = subsample_equal(strong_easy)

    e2h = DatasetDict({"test": weak_hard_test.shuffle(seed=42), "full": weak_hard.shuffle(seed=42)})
    w2s = DatasetDict({"test": strong_easy_test.shuffle(seed=42), "full": strong_easy.shuffle(seed=42)})
    ho = DatasetDict({"test": weak_easy_test.shuffle(seed=42), "full": weak_easy.shuffle(seed=42)})

    e2h = e2h.map(_remove_all_md, num_proc=NUM_WORKERS, desc="Removing markdown formatting")
    w2s = w2s.map(_remove_all_md, num_proc=NUM_WORKERS, desc="Removing markdown formatting")
    ho = ho.map(_remove_all_md, num_proc=NUM_WORKERS, desc="Removing markdown formatting")

    ho.push_to_hub("wetsoledrysoul/Heldout-Set", private=True, max_shard_size="5GB", commit_message="Modify evaluation to lists instead of pairs")
    w2s.push_to_hub("wetsoledrysoul/RQ1-Set", private=True, max_shard_size="5GB", commit_message="Modify evaluation to lists instead of pairs")
    e2h.push_to_hub("wetsoledrysoul/RQ2-Set", private=True, max_shard_size="5GB", commit_message="Modify evaluation to lists instead of pairs")


if __name__ == "__main__":
    main()
