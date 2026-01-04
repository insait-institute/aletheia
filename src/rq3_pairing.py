import os
import random
from itertools import combinations

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
MAX_COMBOS_PER_LEN = 3  # to limit dataset size


def _remove_md(code, lang):
    return code.split(f"```{lang}")[-1].split("```")[0].strip()


def _remove_all_md(example):
    example["candidates"] = [_remove_md(c, example["language"]) for c in example["candidates"]]
    return example


# def add_results_to_completions(example, dct):
#     example["completions"] = dct[example["id"]][example["generator"]][0]
#     example["description"] = dct[example["id"]][example["generator"]][1]
#     return example


def _add_chosen_rejected(example):
    example["chosen"] = example["candidates"][example["chosen_position"]]
    example["rejected"] = example["candidates"][~example["chosen_position"]]
    example["chosen_pass_rate"] = example["pass_rate"][example["chosen_position"]]
    example["rejected_pass_rate"] = example["pass_rate"][~example["chosen_position"]]
    return example


# def pass_rates_and_filter(example):
#     example["pass_rates"] = [x / (x + y) for (x, y) in zip(example["num_passed"], example["num_failed"])]
#     tokenized_completions = tokenizer(example["completions"], padding=False, truncation=False)["input_ids"]

#     example["completions"] = [x for x, z in zip(example["completions"], tokenized_completions) if len(z) <= 800]
#     example["pass_rates"] = [y for y, z in zip(example["pass_rates"], tokenized_completions) if len(z) <= 800]
#     example["num_surviving"] = len(example["completions"])
#     return example


# def main():
#     # rb = load_dataset("wetsoledrysoul/ruby_execs")["train"]
#     # rs = load_dataset("wetsoledrysoul/rust_execs")["train"]
#     # csharp = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/*.parquet")["train"]
#     # d = load_dataset("wetsoledrysoul/d_execs")["train"]
#     rs = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/rust_veryveryfied.parquet")["train"]
#     rb = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/ruby_veryveryfied.parquet")["train"]
#     csharp = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/csharp_veryveryfied.parquet")["train"]
#     d = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/d_veryveryfied.parquet")["train"]
#     data = concatenate_datasets([rs, rb, csharp, d])
#     og = load_dataset("wetsoledrysoul/Heldout-Set", split="full")
#     pid_to_desc = {ex["prompt_id"]: ex["query"] for ex in og}
#     # 90% of the completions are less than 800 tokens. Set that as the limit
#     answer_list = ["A", "B", "C", "D", "E"]
#     paired_data = []
#     for example in tqdm(data, desc="Creating listwise eval sets", total=len(data)):
#         pass_rates = example["pass_rate"]
#         codes = example["completions"]
#         if not codes:
#             print("no codes at all")
#             continue
#         chosen = [x for x, y in zip(codes, pass_rates) if y == 1.0]
#         if not len(chosen):
#             continue

#         # Cluster codes with same pass rates together to maintain list diversity
#         ratio_to_codes = {}
#         for code, ratio in zip(codes, pass_rates):
#             if ratio == 1.0:
#                 continue
#             if ratio not in ratio_to_codes:
#                 ratio_to_codes[ratio] = []
#             ratio_to_codes[ratio].append(code)

#         easy_ratios = [ratio for ratio in ratio_to_codes.keys() if ratio <= 0.5]
#         maxlen_easy = min(len(easy_ratios), 4) + 1

#         if maxlen_easy >= 2:
#             for ith_len in range(1, maxlen_easy):
#                 # create all possible combinations of easy ratios of length ith_len
#                 all_ratio_combos = [list(x) for x in combinations(easy_ratios, ith_len)]


#                 MAX_COMBOS = 20000000
#                 selected_combos = random.sample(all_ratio_combos, min(MAX_COMBOS, len(all_ratio_combos)))
#                 # for each combination, select one code from each ratio bucket and combine with one chosen code
#                 for ratios in selected_combos:
#                     code_buckets = [chosen] + [ratio_to_codes[ratio] for ratio in ratios]
#                     base_ratios = [1.0] + [x for x in ratios]
#                     all_code_combos = list(product(*code_buckets))
#                     MAX_PRODS = 20000000
#                     all_code_combos = random.sample(all_code_combos, min(len(all_code_combos), MAX_PRODS))
#                     for code_combo in all_code_combos:
#                         paired = list(zip(code_combo, base_ratios))
#                         random.shuffle(paired)
#                         codes_list, pr_list = zip(*paired)
#                         paired_data.append(
#                             {
#                                 "prompt_id": example["prompt_id"],
#                                 "query": pid_to_desc[example["prompt_id"]],
#                                 "generator": example["generator"],
#                                 "language": example["language"],
#                                 "num_candidates": len(codes_list),
#                                 "candidates": codes_list,
#                                 "chosen_position": pr_list.index(1.0),
#                                 "chosen_answer": answer_list[pr_list.index(1.0)],
#                                 "pass_rates": pr_list,
#                                 "easy_hard": "easy",
#                                 "weak_strong": "weak",
#                             }
#                         )
def main():
    # d = load_dataset("wetsoledrysoul/d_execs")["train"]
    rs = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/rust_veryfied.parquet")["train"]
    rb = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/ruby_veryfied.parquet")["train"]
    csharp = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/csharp_veryfied.parquet")["train"]
    d = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/d_veryfied.parquet")["train"]
    data = concatenate_datasets([rs, rb, csharp, d])
    og = load_dataset("wetsoledrysoul/Heldout-Set", split="full")
    pid_to_desc = {ex["prompt_id"]: ex["query"] for ex in og}
    # 90% of the completions are less than 800 tokens. Set that as the limit
    answer_list = ["A", "B", "C", "D", "E"]
    paired_data = []
    for example in tqdm(data, desc="Creating listwise eval sets", total=len(data)):
        pass_rates = example["pass_rate"]
        codes = example["completions"]
        if not codes:
            continue
        chosen = list(set([x for x, y in zip(codes, pass_rates) if y == 1.0]))
        if not len(chosen):
            continue

        rejected = []
        rejected_prs = []
        for code, pr in zip(codes, pass_rates):
            if pr != 1.0 and code not in rejected:
                rejected.append(code)
                rejected_prs.append(pr)

        if not len(rejected):
            continue
        for ith_len in range(1, min(len(rejected), 4) + 1):
            rejected_with_prs = list(zip(rejected, rejected_prs))
            for i, combo_of_pairs in enumerate(combinations(rejected_with_prs, ith_len)):
                if i > 1000 and example["language"] != "ruby":
                    break
                rejected_codes = [pair[0] for pair in combo_of_pairs]
                rejected_pr_list = [pair[1] for pair in combo_of_pairs]
                for chosen_code in chosen:
                    code_combo = [chosen_code] + rejected_codes
                    pr_combo = [1.0] + rejected_pr_list
                    paired = list(zip(code_combo, pr_combo))
                    random.shuffle(paired)
                    codes_list, pr_list = zip(*paired)
                    paired_data.append(
                        {
                            "prompt_id": example["prompt_id"],
                            "query": pid_to_desc[example["prompt_id"]],
                            "generator": example["generator"],
                            "language": example["language"],
                            "num_candidates": len(code_combo),
                            "candidates": codes_list,
                            "chosen_position": codes_list.index(chosen_code),
                            "chosen_answer": answer_list[codes_list.index(chosen_code)],
                            "pass_rates": pr_list,
                            "easy_hard": "easy",
                            "weak_strong": "weak",
                        }
                    )

    def subsample_equal(ds, n=250):
        ds = ds.to_pandas()
        ds = ds.groupby(["num_candidates", "language"]).apply(lambda x: x.sample(n=min(len(x), n), random_state=42)).reset_index(drop=True)
        return Dataset.from_pandas(ds)

    rq3_full = Dataset.from_list(paired_data).shuffle(seed=42)
    rq3_full = rq3_full.add_column("id", [f"x_ling_{i}" for i, _ in enumerate(rq3_full)])
    breakpoint()
    rq3_test = subsample_equal(rq3_full, n=200)
    final = DatasetDict({"test": rq3_test, "full": rq3_full})
    final = final.map(_remove_all_md, num_proc=NUM_WORKERS, desc="Removing markdown formatting")
    final.push_to_hub("wetsoledrysoul/Mehnat-Set", private=True, max_shard_size="5GB", commit_message="add questions")


if __name__ == "__main__":
    main()
    # select a random sample from each ratio bucket
