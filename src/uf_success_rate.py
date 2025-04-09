import ast
import logging
import random
import re
from pathlib import Path

import datasets
import hydra
import openai
import torch
from tenacity import (
    Retrying,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
    wait_fixed,
)
from tqdm import tqdm
from vllm import LLM, SamplingParams

from configs.uf_success_rate_config import Config

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """<|im_start|>system
Your role is to evaluate text quality based on given criteria.
You'll receive a user input and two assistant responses. 
You must choose one of the responses that better fits the user's request.
You must evaluate the alignment between the output and the user's intent.
Assess each assistant's understanding of the task goal (the intended outcome) and restrictions (text styles, formats, designated methods, etc).
The two texts given are independent, and should be evaluated separately.
<|im_end|>

<|im_start|>user
Here is the user's input and the two assistant responses. Evaluate them and choose the one that better fits the user's request.
<user_input>
{user_input}
</user_input>

<response_A>
{response_A}
</response_A>

<response_B>
{response_B}
</response_B>

Output your answer ONLY as a markdown json block in the following format:
```json
{{
    "reasoning": "<your reasoning for selecting a response here>",
    "preferred_response": "[[A]] or [[B]], where refers [[A]] is the first response and [[B]] is the second response",
}}
```
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


def clean_llm_output(output: str) -> tuple[dict[str, str], str]:
    if "</think>" in output:
        rationale = re.match(r"^(.*?)</think>", output, re.DOTALL)
        rationale = rationale.group(1).replace("<think>", "").strip() if rationale else ""
    else:
        rationale = ""
    output = re.sub(r"^.*?</think>", "", output, re.DOTALL).strip()
    match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
    try:
        match = ast.literal_eval(match.group(1).strip())
    except Exception:
        match = {}
    return (match, rationale)


def fill_prompt(instruction: str, resp1: str, resp2: str) -> tuple[str, int]:
    if random.random() < 0.5:
        # Randomly swap code1 and code2
        return PROMPT_TEMPLATE.format(user_input=instruction, response_A=resp1, response_B=resp2), 0
    else:
        return PROMPT_TEMPLATE.format(user_input=instruction, response_A=resp1, response_B=resp2), 1


def get_content(responses: list[dict]) -> str:
    return [dct["content"] for dct in responses if dct["role"] == "assistant"][0]


@hydra.main(version_base=None, config_name="uf_success_rate_config")
def main(cfg: Config) -> None:
    dataset = datasets.load_dataset(cfg.db.data.name, split=cfg.db.data.split)
    dataset = dataset.shuffle(seed=cfg.db.data.seed).select(range(cfg.db.data.num_samples))
    log.info(f"Loaded {len(dataset)} samples from {cfg.db.data.name}.")
    log.info(f"Using model: {cfg.db.model.name}")
    log.info(f"Using sampling params: {cfg.db.sparams}")
    avg_score_difference = sum([x - y for x, y in zip(dataset["score_chosen"], dataset["score_rejected"])]) / len(dataset)
    log.info(f"Average score difference: {avg_score_difference:.2f}")
    # average_score_difference = dataset["score_chosen"] - dataset["score_rejected"]

    fill_results = [fill_prompt(x["prompt"], get_content(x["chosen"]), get_content(x["rejected"])) for x in dataset]
    prompts, swapped = zip(*fill_results)
    dataset = dataset.add_column("swapped", swapped)

    prompts = [_prompt_to_chatml(prompt) for prompt in prompts]
    log.info(f"Generated {len(prompts)} prompts.")

    if cfg.db.model.name in ["deepseek-ai/DeepSeek-R1"]:
        responses = []
        if cfg.db.model.service == "azure_openai":
            client = openai.AzureOpenAI(
                api_key=cfg.db.model.api_key,
                azure_endpoint=cfg.db.model.api_base,
                api_version="2024-12-01-preview",
            )
        else:
            client = openai.OpenAI(
                api_key=cfg.db.model.api_key,
                base_url=cfg.db.model.api_base,
            )

        if cfg.db.retries.exponential_backoff:
            wait_config = wait_exponential(
                min=cfg.db.retries.retry_interval,
                max=cfg.db.retries.max_retry_interval,
                multiplier=2,
            )
        else:
            wait_config = wait_fixed(cfg.db.retries.retry_interval)

        for prompt in tqdm(prompts, total=len(prompts), desc="Generating responses"):
            for attempt in Retrying(
                retry=retry_if_exception_type(openai.RateLimitError)
                | retry_if_exception_type(openai.APIError)
                | retry_if_exception_type(openai.OpenAIError)
                | retry_if_result(lambda result: cfg.db.retries.retry_when_blank and any([r.message.content == "" for r in result.choices])),
                stop=stop_after_attempt(cfg.db.retries.max_retries) | stop_after_delay(cfg.db.retries.retry_timeout),
                wait=wait_config,
            ):
                with attempt:
                    response = client.chat.completions.create(
                        messages=prompt,
                        model=cfg.db.model.name,
                        n=cfg.db.sparams.n,
                        temperature=cfg.db.sparams.temperature,
                        max_tokens=cfg.db.sparams.max_tokens,
                        seed=cfg.db.sparams.seed,
                    )
                    responses.extend([choice.message.content for choice in response.choices])
    else:
        llm = LLM(
            cfg.db.model.name,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        sampling_params = SamplingParams(
            temperature=cfg.db.sparams.temperature,
            max_tokens=cfg.db.sparams.max_tokens,
            n=cfg.db.sparams.n,
            seed=cfg.db.sparams.seed,
        )
        responses = llm.chat(prompts, sampling_params)
        responses = [x.outputs[0].text for x in responses]

    log.info(f"Generated {len(responses)} responses.")

    responses = [clean_llm_output(response) for response in responses]
    gen_thoughts = [response[0] for response in responses]
    gen_answers = [response[1].get("preferred_response", None) for response in responses]
    gen_cot = [response[1].get("reasoning", None) for response in responses]
    dataset = dataset.add_column("gen_thoughts", gen_thoughts)
    dataset = dataset.add_column("gen_cot", gen_cot)
    dataset = dataset.add_column("gen_answer", gen_answers)

    correct_answers = ["[[B]]" if swapped else "[[A]]" for swapped in dataset["swapped"]]
    correct = [verdict == answer for verdict, answer in zip(gen_answers, correct_answers)]
    accuracy = sum(correct) / len(correct)
    log.info(f"Model: {cfg.db.model.name}")
    log.info(f"Accuracy: {accuracy:.2%}")
    log.info(f"Correct answers: {sum(correct)}")
    log.info(f"Total answers: {len(correct)}")
    log.info(f"Number of swaps: {sum(dataset['swapped'])}")
    log.info("Completed successfully.")
    model_short_name = cfg.db.model.name.split("/")[-1].lower().replace("-", "_")
    dataset_short_name = cfg.db.data.name.split("/")[-1].lower().replace("-", "_")
    output_dir = Path(__file__).parent / f"outputs/rationales_{dataset_short_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_dir / f"{model_short_name}.csv")


if __name__ == "__main__":
    main()
