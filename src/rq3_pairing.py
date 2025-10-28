import os
import random

from datasets import Dataset, concatenate_datasets, load_dataset
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


def main():
    js = load_dataset("wetsoledrysoul/javascript_execs")["train"]
    rb = load_dataset("wetsoledrysoul/ruby_execs")["train"]
    rs = load_dataset("wetsoledrysoul/rust_execs")["train"]
    go = load_dataset("wetsoledrysoul/go_execs")["train"]
    data = concatenate_datasets([js, rb, rs, go])
    og = load_dataset("CodeShield/CerebRM-Dataset")["test_weak_easy"]
    pid_to_desc = {ex["prompt_id"]: ex["query"] for ex in og}
    # 90% of the completions are less than 800 tokens. Set that as the limit
    answer_list = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"]
    paired_data = []
    for example in data:
        pass_rates = example["pass_rate"]
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

        if len(easy_ratios):
            selected_ratios = random.sample(easy_ratios, min(MAX_COMBOS_PER_LEN, len(easy_ratios)))
            for ratio in selected_ratios:
                codes_list = [random.sample(chosen, 1)[0], random.sample(ratio_to_codes[ratio], 1)[0]]
                pr_list = [1.0, ratio]
                paired = list(zip(codes_list, pr_list))
                random.shuffle(paired)
                codes_list, pr_list = zip(*paired)

                # Add to appropriate list based on model type
                paired_data.append(
                    {
                        "prompt_id": example["prompt_id"],
                        "id": example["id"],
                        "generator": example["generator"],
                        "language": example["language"],
                        "candidates": codes_list,
                        "chosen_position": pr_list.index(1.0),
                        "chosen_answer": answer_list[pr_list.index(1.0)],
                        "pass_rate": pr_list,
                        "query": pid_to_desc[example["prompt_id"]],
                    }
                )

    paired_data = Dataset.from_list(paired_data).shuffle(seed=42)
    paired_data = paired_data.map(_remove_all_md, num_proc=NUM_WORKERS, desc="Removing markdown formatting")
    paired_data = paired_data.map(
        _add_chosen_rejected,
        num_proc=NUM_WORKERS,
        desc="Converting to RB format",
        remove_columns=[
            "chosen_position",
            "chosen_answer",
            "pass_rate",
            "candidates",
        ],
    )
    paired_data.push_to_hub("wetsoledrysoul/RQ3-Set", private=True, max_shard_size="5GB", commit_message="add questions")
    paired_data.push_to_hub("CodeShield/RQ3-Set", private=True, max_shard_size="5GB", commit_message="add questions")


if __name__ == "__main__":
    main()
    # select a random sample from each ratio bucket
