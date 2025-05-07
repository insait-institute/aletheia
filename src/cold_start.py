import logging
import os
import sys
from collections import Counter

import numpy as np
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
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


def construct_messages(example):
    example["answer"] = example["answer"].replace("think>", "reason>").replace("answer>", "solution>")
    example["messages"] = CHATML.format(question=example["question"], answer=example["answer"])
    example["num_tokens"] = num_tokens_from_string(example["messages"])
    return example


def language_filter(example):
    english_confidence = detector.compute_language_confidence(example["messages"], Language.ENGLISH)
    return english_confidence >= 0.99


def length_filter(example):
    return example["num_tokens"] <= 4096


def variabilty_filter(example):
    return example["variabilty"] >= 0.05


def correctness_filter(example):
    if example["category"] == "science":
        return example["verify_score"] > 4.99
    elif example["category"] == "other":
        return example["verify_score"] > 0.7
    return example["verify_score"] > 0.99


def main(model_name):
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
    for category in ["code", "math", "instruction follow", "science", "other"]:
        category_ds = []
        for split in tqdm(splits[category], desc=f"Processing category: {category}", total=len(splits[category])):
            ds = load_dataset("a-m-team/AM-DeepSeek-Distilled-40M", data_files={"train": split}, split="train", features=features)
            ds = ds.map(construct_messages, remove_columns=["question", "answer", "answer_source", "ground_truth", "test_case", "instruction_constrain", "ppl"], num_proc=os.cpu_count())
            ds = ds.add_column("query_id", [i for i in range(len(ds))])
            category_ds.append(ds)
        category_ds = concatenate_datasets(category_ds)

        query_scores = {}
        for example in category_ds:
            query_id = example["query_id"]
            if query_id not in query_scores:
                query_scores[query_id] = []
            query_scores[query_id].append(example["verify_score"])

        variability_map = {}
        for query_id, scores in query_scores.items():
            scores_array = np.array(scores)
            mean = np.mean(scores_array)
            if mean > 0:
                variability_map[query_id] = np.std(scores_array) / mean
            else:
                variability_map[query_id] = 0

        category_ds = category_ds.map(lambda example: {"variability": variability_map.get(example["query_id"], 0)}, num_proc=os.cpu_count())

        ds = ds.filter(lambda x: length_filter(x) and correctness_filter(x) and language_filter(x), num_proc=os.cpu_count())
        full_dataset.append(ds)

    full_dataset = concatenate_datasets(full_dataset)
    full_dataset.save_to_disk(f"../data/cold_start_predupe_{model_name}")
    log.info(f"Total examples after filtering: {len(full_dataset)}")
    log.info(f"Distribution of categories: {Counter(full_dataset['category'])}")
    log.info(f"Columns in the dataset: {full_dataset.column_names}")


if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)
