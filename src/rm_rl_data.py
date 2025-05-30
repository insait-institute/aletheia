import os
from typing import Dict, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

GENRM_INSTRUCTION = "Is this response {aspect}? Answer with a 'Yes' or 'No'."

ASPECT_DICT = {
    0: "readable and maintainable",
    1: "runtime efficient",
    2: "secure",
    3: "functionally correct",
    4: "memory efficient",
    5: "helpful",
    6: "harmless",
}

GENRM_TEMPLATE_NOSYS = """
<|im_start|>user
{input}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>
<|im_start|>user
{genrm_instruction}
<|im_end|>
<|im_start|>assistant
<reason>
"""


def get_genrm_data(example: Dict, col: str) -> Dict:
    aspect = ASPECT_DICT[example["aspect"]]
    genrm_instruction = GENRM_INSTRUCTION.format(aspect=aspect)
    if example["system"]:
        PROMPT_TEMPLATE = f"<|im_start|>system\n{example['system']}\n<|im_end|>\n{GENRM_TEMPLATE_NOSYS}"
    else:
        PROMPT_TEMPLATE = GENRM_TEMPLATE_NOSYS
    example["prompt"] = PROMPT_TEMPLATE.format(
        input=example["input"],
        output=example[col],
        genrm_instruction=genrm_instruction,
    )
    example["verdict"] = "Yes" if col == "chosen" else "No"
    return example


def process_data(data: Dataset, col: Optional[str] = None) -> Dataset:
    data = data.map(
        get_genrm_data,
        fn_kwargs={"col": col},
        remove_columns=["chosen", "rejected", "system"],
        num_proc=os.cpu_count(),
    )

    return data


def main():
    general_pref = load_dataset("CodeShield/General-Preference")
    commit_pref = load_dataset("CodeShield/CPE_Markdownized")
    # for cp_col in commit_pref["train"].column_names:
    #     commit_pref = commit_pref.cast_column(cp_col, general_pref["train"].features[cp_col])

    # commit_pref.push_to_hub(
    #     repo_id="CodeShield/CPE_Markdownized",
    #     private=True,
    # )
    data_train = concatenate_datasets([general_pref["train"], commit_pref["train"]])
    data_test = concatenate_datasets([general_pref["test"], commit_pref["test"]])

    data_train = concatenate_datasets(
        [
            process_data(data_train, "chosen"),
            process_data(data_train, "rejected"),
        ]
    ).shuffle(seed=42)
    data_test = concatenate_datasets(
        [
            process_data(data_test, "chosen"),
            process_data(data_test, "rejected"),
        ]
    ).shuffle(seed=42)

    final_data = DatasetDict(
        {
            "train": data_train,
            "test": data_test,
        }
    )
    final_data.push_to_hub(
        repo_id="CodeShield/RLData_RM",
        private=True,
    )
    breakpoint()


if __name__ == "__main__":
    main()
