import logging
from typing import Any, Dict

import datasets
import hydra
import torch
from vllm import LLM, SamplingParams

from configs.markdownize_dataset_config import MDConfig

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

MD_TEMPLATE = """<|im_start|>system
You are a helpful AI assistant. You will be provided a coding question (indicated by <query>...</query>) and a correct but poorly formatted output (indicated by <output>...</output>). Your task is to convert the output into a well-explained and formatted response. All the code you output must be in the markdown format:
```<language>
<code>
```
Explain the code in detail either before or after outputting it. You are not allowed to change the code in any way. You must only format the output and add explanations. Do not add any additional information or context outside of the code and its explanation.
Here are some characteristics of a good markdown response:
- Your response must use proper markdown code blocks for code snippets.
- You CAN NOT split the given code into multiple code blocks.
- You must provide a clear and concise explanation of the code.
- Your response must be in English.
- Your response must directly answer the user's query.
<|im_end|>

<|im_start|>user
<query>
{query}
</query>

<output>
{output}
</output>

<|im_end|>
"""


def create_prompt(example: Dict[str, Any], col: str) -> Dict[str, Any]:
    example["prompt"] = MD_TEMPLATE.format(query=example["input"], output=example[col])
    return example


def run_llm(cfg: MDConfig, prompts: str):
    llm = LLM(cfg.model.name, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        temperature=cfg.sparams.temperature,
        max_tokens=cfg.sparams.max_tokens,
        n=cfg.sparams.n,
        seed=cfg.sparams.seed,
    )
    responses = llm.chat(prompts, sampling_params)
    if cfg.sparams.n == 1:
        return [x.outputs[0].text for x in responses]
    return [[y.text for y in x.outputs] for x in responses]


@hydra.main(version_base=None, config_name="markdownize_dataset_config")
def markdownize(cfg: MDConfig):
    data = datasets.load_dataset(cfg.data.name)
    for split in data.keys():
        for col in [cfg.data.chosen_col, cfg.data.rejected_col]:
            log.info(f"Processing column: {col}")

            data[split] = data[split].map(lambda x: create_prompt(x, col))
            log.info(f"Created {len(data[split])} prompts")
            log.info(f"Sample prompt for {col}:\n{data[split]['prompt'][0]}")

            md_responses = run_llm(cfg, data[split]["prompt"])
            data[split].add_column(f"{col}_md", md_responses)
    ds_short_name = cfg.data.name.split("/")[-1]
    data.save_to_disk(f"markdownized_{ds_short_name}")


if __name__ == "__main__":
    markdownize()
