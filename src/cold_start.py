# import datasets


# class Dataset:
#     def __init__(self):
#         self.name = None
#         self.config = None
#         self.messages_col = None

#     def load(self):
#         if self.config:
#             dataset = datasets.load_dataset(self.name, self.config)
#         else:
#             dataset = datasets.load_dataset(self.name)
#         return dataset

#     def process(self):
#         data = self.raw_data.map(self._process_example, batched=True)
#         return data

#     def create_sft_dataset(self):
#         self.data = self.processed_data.select_columns([self.message_col, "source"])


# class KodCode(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.name = "KodCode/KodCode-V1-SFT-R1"
#         self.messages_col = "conversations"

#         self.raw_data = self.load()
#         self.processed_data = self.process()

#     def _process_example(self, example):
#         messages = example[self.messages_col]
#         user_question = messages[0]["value"]
#         assistant_answer = messages[1]["value"]
#         messages = [
#             {"role": "user", "content": user_question},
#             {"role": "assistant", "content": assistant_answer},
#         ]
#         example[self.messages_col] = messages
#         example["source"] = "kodcode"
#         return example


# class CuratedThoughtsOpenR1(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.name = "bethgelab/CuratedThoughts"
#         self.split = "OpenR1-Math-220k-default"
#         self.messages_col = "conversations"

#         self.raw_data = self.load()
#         self.processed_data = self.process()

#     def _process_example(self, example):
#         messages = example[self.messages_col]
#         user_question = messages[0]["value"]
#         assistant_answer = messages[1]["value"]
#         messages = [
#             {"role": "user", "content": user_question},
#             {"role": "assistant", "content": assistant_answer},
#         ]
#         example[self.messages_col] = messages
#         example["source"] = "curatedthoughts_openr1"
#         return example


# class CuratedThoughtsOpenThoughts(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.name = "bethgelab/CuratedThoughts"
#         self.split = "OpenThoughts-114k-math-default"
#         self.messages_col = "conversations"

#         self.raw_data = self.load()
#         self.processed_data = self.process()

#     def _process_example(self, example):
#         messages = example[self.messages_col]
#         user_question = messages[0]["value"]
#         assistant_answer = messages[1]["value"]
#         messages = [
#             {"role": "user", "content": user_question},
#             {"role": "assistant", "content": assistant_answer},
#         ]
#         example[self.messages_col] = messages
#         example["source"] = "curatedthoughts_openthoughts"
#         return example


# def cold_start():
#     # Load kodcode, curatedthoughts
#     kodcode = KodCode()
#     curated_thoughts_openr1 = CuratedThoughtsOpenR1()
#     curated_thoughts_openthoughts = CuratedThoughtsOpenThoughts()


# if __name__ == "__main__":
#     cold_start()


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


def simplify_messages(example, split):
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
    data = data.map(simplify_messages, fn_kwargs={"split": split})
    final_datasets.append(data)

# Merge datasets
merged_dataset = concatenate_datasets(final_datasets)

data = DatasetDict({"train": merged_dataset})
data.push_to_hub("wetsoledrysoul/cold_start_data", private=True)
