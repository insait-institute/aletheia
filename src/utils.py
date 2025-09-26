import logging
import re
from typing import Dict, List, Union

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def run_inference(
    prompts: List[Dict[str, str]],
    llm: Union[str, LLM],
    tokenizer: AutoTokenizer = None,
    enable_thinking=False,
    tp_size=1,
    temperature=1.0,
    max_tokens=4096,
    n=1,
    skip_special_tokens=True,
    gpu_memory_utilization=0.95,
    **kwargs,
):
    if isinstance(llm, str):
        tokenizer = AutoTokenizer.from_pretrained(llm, trust_remote_code=True)
        llm = LLM(
            model=llm,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided if llm is an instance of LLM")
    prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        skip_special_tokens=skip_special_tokens,
        **kwargs,
    )
    log.info(f"Running inference on {len(prompts)} prompts")
    responses = llm.generate(prompts, sampling_params)
    return responses


def extract_solution(output):
    solution = output
    if "<answer>" in output:
        solution = output.split("<answer>")[-1].split("</answer>")[0]
    elif "<solution>" in output:
        solution = output.split("<solution>")[-1].split("</solution>")[0]
    elif "</think>" in output:
        solution = output.split("</think>")[-1]
    solution = re.sub(r"\{+", "{", solution)
    solution = re.sub(r"\}+", "}", solution)
    pattern = r"\\boxed\{(.*?)\}"
    match = re.search(pattern, solution, re.DOTALL)
    answer = match.group(1).strip() if match else "Error"
    return answer
