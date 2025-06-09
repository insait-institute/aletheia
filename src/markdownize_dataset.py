import logging
import os
import re
from pathlib import Path
from typing import Any, Dict

import hydra
import polars as pl
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from vllm import LLM, SamplingParams

from configs.markdownize_dataset_config import MDConfig

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)
LANG_MAP = {
    "0": "js",
    "1": "python",
    "2": "ruby",
    "3": "cpp",
    "4": "c",
    "5": "go",
    "6": "java",
    "7": "csharp",
    "8": None,
}

MD_TEMPLATE = """<|im_start|>system 
You are the founder and CEO of a successful startup that provides solutions to coding questions posed by your client, Alex. You work with your co-founder and an excellent software developer, Bob who comes up with answers to your client's coding questions. Due to excessive workload, Bob can only write the code that solves Alex's question, but is unable to provide an explanation of how his works or his thought process for arriving at the final solution. You are very strict about the quality of the answers provided to your clients, especially for the quality of the code and the explanations provided in the answers.
<|im_end|>

<|im_start|>user
Your client, Alex, has asked a coding question (ALEX_QUESTION) and your colleague, Bob, has provided a correct, efficient and secure response (BOB_RESPONSE). You must write write a final response that can be sent to Alex, which will contain Bob's code at the location of a placeholder token, [BOB_CODE].

Here are the guidelines that you must follow. Failure to follow any of these guidelines is considered a poor response and you will lose Alex as a client.

1. You must include a placeholder token, [BOB_CODE] at ONE location in your response. This token will be replaced with Bob's code when the response is sent to Alex. Typically, this should be within the first few lines of your response. 

2. You must include EXACTLY ONE [BOB_CODE] token. Any different number of occurrences of the token will be considered a mistake and you will lose Alex as a client.

3. DO NOT output any code snippets at all. A response containing code will be immediately be rejected by Alex.

4. DO NOT refer to Bob or Alex by name.

5. Your response should be a reply to Alex's question, not a simple explanation of Bob's code. A good template to follow is to start with words like "Certainly!", "Of course!", etc. and provide a brief introduction to your thought process about Alex's question. Then you should include [BOB_CODE] and follow it with a detailed explanation of the Bob's code.

6. Explain the code in detail, including the logic behind it. Your explanation must be in English. The explanation should be clear and should give Alex the confidence that the code provided is correct, efficient and secure.

7. Avoid using emojis and special unicode characters unless part of the code. You are a professional and your response should reflect that.


Your response must be enclosed within <FINAL_ANSWER> and </FINAL_ANSWER> tags as shown below:

<FINAL_ANSWER>
{{Your response to Alex's question containing a single [BOB_CODE] token, goes here.}}
</FINAL_ANSWER>

Here is Alex's question and Bob's response that you must format. The question is enclosed within <ALEX_QUESTION> and </ALEX_QUESTION>, and Bob's response is enclosed within <BOB_RESPONSE> and </BOB_RESPONSE>. Reply only in the format described above, and do not include any additional text or explanations.

<ALEX_QUESTION>

{query}

</ALEX_QUESTION>

<BOB_RESPONSE>

{output}

</BOB_RESPONSE>

<|im_end|>
"""


def _prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>") -> dict[str, str]:
    prompt = prompt.strip()
    message = []
    for p in prompt.split(start_token)[1:]:
        newline_splitted = p.split("\n", 1)
        role = newline_splitted[0].strip()
        content = newline_splitted[1].split(end_token, 1)[0].strip()

        if role.startswith("system") and role != "system":
            other_params = _string_to_dict(role.split("system", 1)[-1])
            role = "system"
        else:
            other_params = dict()

        message.append(dict(content=content, role=role, **other_params))

    return message


def _string_to_dict(to_convert: str) -> dict[str, str]:
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def create_prompt(example: Dict[str, Any], col: str) -> Dict[str, Any]:
    example["prompt"] = MD_TEMPLATE.format(query=example["input"], output=example[col])
    example["prompt"] = _prompt_to_chatml(example["prompt"])
    return example


def clean_llm_output(output: str) -> str:
    output = re.sub(r"^(.*?)</think>", "", output, flags=re.DOTALL)
    output = output.strip()
    return output


def extract_final_answer(output: str) -> str:
    output = clean_llm_output(output)
    final_answer = output.split("<FINAL_ANSWER>")[-1].split(0).strip("</FINAL_ANSWER>")
    return final_answer.group(1).strip() if final_answer else ""


def run_llm(llm: LLM, sampling_params: SamplingParams, prompts: str):
    responses = llm.chat(prompts, sampling_params)
    if sampling_params.n == 1:
        return [x.outputs[0].text for x in responses]
    return [[y.text for y in x.outputs] for x in responses]


def markdownize(cfg: MDConfig):
    llm = LLM(cfg.model.name, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count(), max_model_len=cfg.model.max_model_len)
    sampling_params = SamplingParams(
        temperature=cfg.sparams.temperature,
        max_tokens=cfg.sparams.max_tokens,
        n=cfg.sparams.n,
        seed=cfg.sparams.seed,
    )
    data = load_dataset(cfg.data.name, trust_remote_code=True)

    log.info(f"Processing column: {cfg.data.col}, cfg.data.split: {cfg.data.split}")
    # data[cfg.data.split] = data[cfg.data.split].select(range(100))
    data[cfg.data.split] = data[cfg.data.split].map(create_prompt, fn_kwargs={"col": cfg.data.col}, num_proc=os.cpu_count(), desc="Creating prompts for markdownization")
    log.info(f"Created {len(data[cfg.data.split])} prompts")
    md_responses = run_llm(llm, sampling_params, data[cfg.data.split]["prompt"])
    final_answers = [extract_final_answer(x) for x in md_responses]

    data[cfg.data.split] = data[cfg.data.split].add_column(f"{cfg.data.col}_md", final_answers)
    data[cfg.data.split] = data[cfg.data.split].remove_columns(["prompt"])
    ds_short_name = cfg.data.name.split("/")[-1]
    save_dir = Path(__file__).parent.parent / f"MD_{ds_short_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving dataset to {save_dir}")
    data[cfg.data.split].save_to_disk(save_dir / f"{cfg.data.split}_{cfg.data.col}", max_shard_size="5GB")
    log.info(f"Dataset markdownized and saved to {save_dir / f'{cfg.data.split}_{cfg.data.col}'}")


def filter_bobcode_by_count(example: Dict) -> bool:
    count = example["output"].count("[BOB_CODE]")
    return count == 1


def replace_bobcode(example: Dict, col: str) -> Dict:
    if example["language"] not in LANG_MAP or example["language"] == 8:
        log.warning(f"Example with unsupported language: {example['language']}")
        replacement = example[col]
    replacement = f"```{LANG_MAP[example['language']]}\n{example[col]}\n```"
    example[col] = example[f"{col}_md"].replace("[BOB_CODE]", replacement)
    return example


def combine(cfg: MDConfig):
    ds_short_name = cfg.data.name.split("/")[-1]
    save_dir = Path(__file__).parent.parent / f"MD_{ds_short_name}"
    try:
        train_chosen = load_from_disk(save_dir / "train_chosen")
        train_rejected = load_from_disk(save_dir / "train_rejected")
        test_chosen = load_from_disk(save_dir / "test_chosen")
        test_rejected = load_from_disk(save_dir / "test_rejected")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found in {save_dir}. Please run the markdownize step first.")
    train_chosen = train_chosen.filter(
        filter_bobcode_by_count,
        num_proc=os.cpu_count(),
        desc="Filtering [BOB_CODE] by count in train_chosen",
    )
    train_rejected = train_rejected.filter(
        filter_bobcode_by_count,
        num_proc=os.cpu_count(),
        desc="Filtering [BOB_CODE] by count in train_rejected",
    )
    test_chosen = test_chosen.filter(
        filter_bobcode_by_count,
        num_proc=os.cpu_count(),
        desc="Filtering [BOB_CODE] by count in test_chosen",
    )
    test_rejected = test_rejected.filter(
        filter_bobcode_by_count,
        num_proc=os.cpu_count(),
        desc="Filtering [BOB_CODE] by count in test_rejected",
    )

    train_chosen = train_chosen.map(
        replace_bobcode,
        num_proc=os.cpu_count(),
        fn_kwargs={"col": "chosen"},
        desc="Replacing [BOB_CODE] in train_chosen",
        remove_columns=["chosen_md"],
    )
    train_rejected = train_rejected.map(
        replace_bobcode,
        num_proc=os.cpu_count(),
        fn_kwargs={"col": "rejected"},
        desc="Replacing [BOB_CODE] in train_rejected",
        remove_columns=["rejected_md"],
    )
    test_chosen = test_chosen.map(
        replace_bobcode,
        num_proc=os.cpu_count(),
        fn_kwargs={"col": "chosen"},
        desc="Replacing [BOB_CODE] in test_chosen",
        remove_columns=["chosen_md"],
    )
    test_rejected = test_rejected.map(
        replace_bobcode,
        num_proc=os.cpu_count(),
        fn_kwargs={"col": "rejected"},
        desc="Replacing [BOB_CODE] in test_rejected",
        remove_columns=["rejected_md"],
    )

    train_chosen = pl.from_arrow(train_chosen.data.table)
    train_rejected = pl.from_arrow(train_rejected.data.table)
    test_chosen = pl.from_arrow(test_chosen.data.table)
    test_rejected = pl.from_arrow(test_rejected.data.table)

    train = pl.concat([train_chosen, train_rejected], how="align")
    test = pl.concat([train_chosen, test_rejected], how="align")

    cpe_md = DatasetDict(
        {
            "train": Dataset(train.to_arrow()),
            "test": Dataset(test.to_arrow()),
        }
    )
    cpe_md
    cpe_md.push_to_hub(
        "CodeShield/CPE_Markdownized",
        private=True,
        max_shard_size="5GB",
    )


@hydra.main(version_base=None, config_name="markdownize_dataset_config")
def main(cfg):
    if cfg.mode == "markdownize":
        markdownize(cfg)
    elif cfg.mode == "combine":
        combine(cfg)
    else:
        raise ValueError(f"Invalid mode: {cfg.mode}. Choose 'markdownize' or 'combine'")


if __name__ == "__main__":
    main()
