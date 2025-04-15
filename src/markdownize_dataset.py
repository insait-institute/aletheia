import logging
import re
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
You are a helpful AI assistant. You will be provided a coding question asked by Alex (within <alex_question>...</alex_question>) and a correct but poorly formatted response provided by a Bob (within <bob_response>...</bob_response>). Your task is to convert the Bob's response into a well-explained and formatted answer to Alex's question. Here are some guidelines you should follow:
- You must pretend to be Bob, and your response should directly reply to Alex's their question, containing Bob's response exactly as is provided.
- Don't just explain the code — make sure your reply answers what Alex asked using helpful, natural language and explain the thought process that may have led Bob to arriving at the code. 
- DO NOT refer to yourself as Bob or the user as Alex. Simply reply as if you are Bob, and the user is Alex.
- Wrap the code block in triple backticks (```) with the appropriate language identifier (e.g., python, javascript, etc.), unless already wrapped.
- Avoid using emojis and special unicode characters unless part of the code.
- DO NOT change the code block in any way — keep the formatting, indentation, comments, and line breaks exactly as they are in Bob's response. You must simply include Bob's code in your response as is, and explain it as if you came up with it.
- Write the explanation only in English.
- DO NOT introduce new code, even if it looks like a "better version".
- DO NOT alter variable names, comments, or output formatting.
- You MUST include Bob's response exactly as it is in your response.
<|im_end|>

<|im_start|>user
<alex_question>
{query}
</alex_question>

<bob_response>
{output}
</bob_response>

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


def run_llm(llm: LLM, sampling_params: SamplingParams, prompts: str):
    responses = llm.chat(prompts, sampling_params)
    if sampling_params.n == 1:
        return [clean_llm_output(x.outputs[0].text) for x in responses]
    return [[clean_llm_output(y.text) for y in x.outputs] for x in responses]


@hydra.main(version_base=None, config_name="markdownize_dataset_config")
def markdownize(cfg: MDConfig):
    llm = LLM(cfg.model.name, trust_remote_code=True, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        temperature=cfg.sparams.temperature,
        max_tokens=cfg.sparams.max_tokens,
        n=cfg.sparams.n,
        seed=cfg.sparams.seed,
    )
    data = datasets.load_dataset(cfg.data.name, trust_remote_code=True)
    for split in data.keys():
        for col in [cfg.data.chosen_col, cfg.data.rejected_col]:
            log.info(f"Processing column: {col}, split: {split}")
            data[split] = data[split].map(lambda x: create_prompt(x, col))
            log.info(f"Created {len(data[split])} prompts")

            md_responses = run_llm(llm, sampling_params, data[split]["prompt"])
            data[split] = data[split].add_column(f"{col}_md", md_responses)
            data[split] = data[split].remove_columns(["prompt"])
    ds_short_name = cfg.data.name.split("/")[-1]
    model_short_name = cfg.model.name.split("/")[-1].lower()
    data.save_to_disk(f"{ds_short_name}-Md-{model_short_name}")


if __name__ == "__main__":
    markdownize()
    log.info("Markdownization completed successfully!")
