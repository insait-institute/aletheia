import logging
import os
import sys
from collections import Counter
from typing import Dict

import pyarrow.compute as pc
import tiktoken
from datasets import Features, Value, concatenate_datasets, load_dataset
from lingua import Language, LanguageDetectorBuilder
from tqdm import tqdm

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHATML = """<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>
"""

languages = [Language.ENGLISH, Language.CHINESE, Language.HINDI, Language.SPANISH, Language.ARABIC, Language.FRENCH, Language.RUSSIAN, Language.GERMAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build()


def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def construct_messages(example: Dict) -> Dict:
    example["answer"] = example["answer"].replace("think>", "reason>").replace("answer>", "solution>")
    example["messages"] = CHATML.format(question=example["question"], answer=example["answer"])
    example["num_tokens"] = num_tokens_from_string(example["messages"])
    return example


def language_filter(example: Dict) -> bool:
    english_confidence = detector.compute_language_confidence(example["messages"], Language.ENGLISH)
    return english_confidence >= 0.99


def length_filter(example: Dict) -> bool:
    return example["num_tokens"] <= 4096


def variabilty_filter(example: Dict) -> bool:
    return example["variabilty"] >= 0.05


def correctness_filter(example: Dict) -> bool:
    if example["category"] == "other":
        return example["verify_score"] > 0.7
    return example["verify_score"] > 0.99


def all_filters(example: Dict) -> bool:
    return length_filter(example) and correctness_filter(example) and language_filter(example)


def main(model_name: str) -> None:
    features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "question_source": Value("string"),
            "answer_source": Value("string"),
            "category": Value("string"),
            "ground_truth": Value("string"),
            "test_case": Value("string"),
            "instruction_constrain": Value("string"),
            "pass_rate_r1": Value("float32"),
            "pass_rate_7b": Value("float32"),
            "pass_rate_1.5b": Value("float32"),
            "verify_score": Value("float32"),
            "ppl": Value("float32"),
            "model_name": Value("string"),
        }
    )

    splits = {
        "code": [f"code_{model_name}_{i}pass.jsonl" for i in range(1, 5)],
        "math": [f"math_{model_name}_{i}pass.jsonl" for i in range(1, 5)],
        "instruction follow": [f"if_{model_name}_{i}pass.jsonl" for i in range(1, 5)],
        "science": [f"science_{model_name}_{i}pass.jsonl" for i in range(1, 5)],
        "other": [f"other_{model_name}_{i}pass.jsonl" for i in range(1, 5)],
    }

    full_dataset = []
    for category in splits:
        category_ds = []
        for split in tqdm(splits[category], desc=f"Processing {category}", total=len(splits[category])):
            ds = load_dataset(
                "a-m-team/AM-DeepSeek-Distilled-40M",
                data_files={"train": split},
                split="train",
                features=features,
            )
            ds = ds.sort("question")
            ds = ds.map(
                construct_messages,
                num_proc=os.cpu_count(),
                desc="Constructing messages",
                remove_columns=["question", "answer", "answer_source", "ground_truth", "test_case", "instruction_constrain", "ppl"],
            )
            ds = ds.add_column("query_id", [i for i in range(len(ds))])
            category_ds.append(ds)
        # Concatenate all splits for this category
        category_ds = concatenate_datasets(category_ds)
        table = category_ds.data.table
        grouped = table.group_by(["query_id"]).aggregate([("verify_score", "mean"), ("verify_score", "stddev")])
        ratio = pc.divide(grouped["verify_score_stddev"], grouped["verify_score_mean"])
        grouped = grouped.append_column("variability", ratio)
        lookup = {
            qid: var
            for qid, var in zip(
                grouped["query_id"].to_pylist(),
                grouped["variability"].to_pylist(),
            )
        }
        category_ds = category_ds.map(
            lambda x: {"variabilty": lookup[x["query_id"]]},
            remove_columns=[],
            num_proc=os.cpu_count(),
            desc="Calculating variability",
        )
        category_ds = category_ds.filter(
            all_filters,
            num_proc=os.cpu_count(),
            desc="Filtering examples by length, correctness, and language",
        )

        full_dataset.append(category_ds)
        log.info(f"Final dataset columns for {category}: {category_ds.column_names}")
    full_dataset = concatenate_datasets(full_dataset)
    full_dataset = full_dataset.select_columns(["messages", "variabilty", "query_id"])
    full_dataset.to_parquet(f"../data/cold_start_with_variability_{model_name}.parquet")
    # full_dataset.save_to_disk(f"../data/cold_start_with_variability_{model_name}")
    log.info(f"Total examples after filtering: {len(full_dataset)}")
    log.info(f"Distribution of categories: {Counter(full_dataset['category'])}")
    log.info(f"Columns in the dataset: {full_dataset.column_names}")


if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)
