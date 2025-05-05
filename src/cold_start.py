import tiktoken
from datasets import DatasetDict, Features, Value, concatenate_datasets, load_dataset


def num_tokens_from_messages(messages):
    encoding = tiktoken.get_encoding("o200k_base")
    tokens_per_message = 3
    tokens_per_name = 1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value is None:
                continue
            num_tokens += len(encoding.encode(value, disallowed_special=()))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def extract_metadata(example, split):
    example["original_source"] = example["messages"][0]["info"]["source"]
    example["messages"] = [{"role": msg["role"], "content": msg["content"]} for msg in example["messages"]]
    example["num_tokens"] = num_tokens_from_messages(example["messages"])
    example["am_source"] = split
    return example


final_datasets = []
features = Features(
    {"messages": [{"role": Value("string"), "content": Value("string"), "info": {"source": Value("string"), "reference_answer": Value("string"), "test_case": Value("string"), "think_content": Value("string"), "answer_content": Value("string")}}]}
)

for split in ["am_0.9M", "am_0.5M"]:
    data = load_dataset("a-m-team/AM-DeepSeek-R1-Distilled-1.4M", split, features=features)["train"]
    data = data.map(extract_metadata, fn_kwargs={"split": split}, num_proc=40)
    final_datasets.append(data)

# Merge datasets
merged_dataset = concatenate_datasets(final_datasets)

data = DatasetDict({"train": merged_dataset})
data.push_to_hub("CodeShield/cold_start_data", private=True)
