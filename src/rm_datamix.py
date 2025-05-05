import os
from typing import Dict, Optional

import hydra
from datasets import Dataset, DatasetDict, concatenate_datasets, interleave_datasets, load_dataset

from configs.rm_datamix_config import Config

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
{genrm_response}
<|im_end|>
"""

SFT_TEMPLATE_NOSYS = """
<|im_start|>user
{input}
<|im_end|>
<|im_start|>assistant
{output}
<|im_end|>
"""


def get_genrm_data(example: Dict, col: str) -> Dict:
    aspect = ASPECT_DICT[example["aspect"]]
    genrm_instruction = GENRM_INSTRUCTION.format(aspect=aspect)
    if example["system"]:
        PROMPT_TEMPLATE = f"<|im_start|>system\n{example['system']}\n<|im_end|>\n{GENRM_TEMPLATE_NOSYS}"
    else:
        PROMPT_TEMPLATE = GENRM_TEMPLATE_NOSYS
    example["messages"] = PROMPT_TEMPLATE.format(
        system=example["system"],
        input=example["input"],
        output=example[col],
        genrm_instruction=genrm_instruction,
        genrm_response="Yes" if col == "chosen" else "No",
    )
    return example


def get_sft_data(example: Dict) -> Dict:
    if example["system"]:
        PROMPT_TEMPLATE = f"<|im_start|>system\n{example['system']}\n<|im_end|>\n{SFT_TEMPLATE_NOSYS}"
    else:
        PROMPT_TEMPLATE = SFT_TEMPLATE_NOSYS
    example["messages"] = PROMPT_TEMPLATE.format(
        system=example["system"],
        input=example["input"],
        output=example["chosen"],
    )
    return example


def process_data(data: Dataset, prompt_type: str, col: Optional[str] = None) -> Dataset:
    if prompt_type == "genrm":
        data = data.map(
            get_genrm_data,
            fn_kwargs={"col": col},
            remove_columns=["chosen", "rejected", "system"],
            num_proc=os.cpu_count(),
        )
    elif prompt_type == "sft":
        data = data.map(
            get_sft_data,
            remove_columns=["chosen", "rejected", "system"],
            num_proc=os.cpu_count(),
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}. Must be 'genrm' or 'sft'")
    data = data.add_column("prompt_type", [prompt_type] * len(data))
    return data


@hydra.main(version_base=None, config_name="rm_datamix_config")
def main(cfg: Config):
    data = load_dataset(cfg.data.name)
    data_train = data["train"]
    data_test = data["test"]
    genrm_train = concatenate_datasets(
        [
            process_data(data_train, "genrm", "chosen"),
            process_data(data_train, "genrm", "rejected"),
        ]
    ).shuffle(seed=cfg.data.seed)
    genrm_test = concatenate_datasets(
        [
            process_data(data_test, "genrm", "chosen"),
            process_data(data_test, "genrm", "rejected"),
        ]
    ).shuffle(seed=cfg.data.seed)

    sft_train = process_data(data_train, "sft").shuffle(seed=cfg.data.seed)
    sft_test = process_data(data_test, "sft").shuffle(seed=cfg.data.seed)

    probabilities = [1.0 / (1.0 + cfg.data._lambda), cfg.data._lambda / (1.0 + cfg.data._lambda)]
    final_data = DatasetDict(
        {
            "train": interleave_datasets([genrm_train, sft_train], probabilities=probabilities, seed=cfg.data.seed),
            "test": interleave_datasets([genrm_test, sft_test], probabilities=probabilities, seed=cfg.data.seed),
        }
    )
    final_data.push_to_hub(
        repo_id=f"{cfg.data.name}_GenRM_{cfg.data._lambda}",
        private=True,
    )
    breakpoint()


if __name__ == "__main__":
    main()
