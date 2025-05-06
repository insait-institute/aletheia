import logging
import os
import sys
from collections import Counter

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

    splits_r1 = [
        "code_r1_1pass.jsonl",
        "code_r1_2pass.jsonl",
        "code_r1_3pass.jsonl",
        "code_r1_4pass.jsonl",
        "math_r1_1pass.jsonl",
        "math_r1_2pass.jsonl",
        "math_r1_3pass.jsonl",
        "math_r1_4pass.jsonl",
        "science_r1_1pass.jsonl",
        "science_r1_2pass.jsonl",
        "science_r1_3pass.jsonl",
        "science_r1_4pass.jsonl",
        "if_r1_1pass.jsonl",
        "if_r1_2pass.jsonl",
        "if_r1_3pass.jsonl",
        "if_r1_4pass.jsonl",
        "other_r1_1pass.jsonl",
        "other_r1_2pass.jsonl",
        "other_r1_3pass.jsonl",
        "other_r1_4pass.jsonl",
    ]

    splits_7b = [x.replace("r1", "7b") for x in splits_r1]
    splits = splits_r1 if model_name == "r1" else splits_7b
    data = []
    for split in tqdm(splits, desc="Processing splits", total=len(splits)):
        ds = load_dataset("a-m-team/AM-DeepSeek-Distilled-40M", data_files={"train": split}, split="train", features=features)
        ds = ds.map(construct_messages, remove_columns=["question", "answer", "answer_source", "ground_truth", "test_case", "instruction_constrain", "ppl"], num_proc=os.cpu_count())
        data_len = len(ds)
        ds = ds.filter(lambda x: x["num_tokens"] <= 4096 and x["verify_score"] == 1, num_proc=os.cpu_count())
        filtered_percent = (data_len - len(ds)) / data_len * 100
        log.info(f"Filtering removed {data_len - len(ds)} examples ({filtered_percent:.2f}%)")
        data.append(ds)

    data = concatenate_datasets(data)
    languages = [Language.ENGLISH, Language.CHINESE, Language.HINDI, Language.SPANISH, Language.ARABIC, Language.FRENCH, Language.RUSSIAN, Language.GERMAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    english_confidence = detector.compute_language_confidence_in_parallel(data["messages"], Language.ENGLISH)
    english_confidence = [i for i, conf in enumerate(english_confidence) if conf >= 0.99]
    data = data.select(english_confidence)

    data.save_to_disk(f"../data/cold_start_predupe_{model_name}")
    log.info(f"Total examples after filtering: {len(data)}")
    log.info(f"Distribution of categories: {Counter(data['category'])}")


if __name__ == "__main__":
    model_name = sys.argv[1]
    main(model_name)
