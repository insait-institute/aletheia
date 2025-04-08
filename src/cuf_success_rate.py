import ast
import logging
import random
import re
from collections import Counter

import datasets
import hydra
import openai
import torch
from omegaconf import OmegaConf
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

from configs.cuf_success_rate_config import Config

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """<|im_start|>system
Please act as an impartial judge and evaluate the quality of the response provided by two AI assistants to the user instruction  displayed below. Your overall evaluation needs to be reflective of the specified evaluation criteria. Be as objective as possible. After providing your rationale, you must pick one of the two AI assistants whose output you prefer. Indicate your preference as "[[A]]" if you prefer the first assistant's response, and "[[B]]" if you prefer the second assistant's response. Do not indicate your preference in any other manner.
<|im_end|>

<|im_start|>user
<user_instruction>
{user_instruction}
</user_instruction>

<response_A>
{code1}
</response_A>

<response_B>
{code2}
</response_B>

<evaluation_criteria>
{evaluation_criteria}
</evaluation_criteria>

Output your answer ONLY as a markdown json block in the following format:
```json
{{
    "rationale": "<your rationale here>",
    "preferred_code": "[[A]] or [[B]], where refers [[A]] is the first response and [[B]] is the second response",
}}
```
<|im_end|>
"""

INSTRUCTION_FOLLOWING = """Evaluate the assistant's fidelity to provided instructions. Assess how accurately the assistant's responses align with user  directives, noting any deviations and their justification.  

Evaluation Criteria:  
    Precision in Following Instructions: Does the assistant adhere to the specifics of the provided instructions?  Justification for Deviations: If deviations occur, are they justified by critical necessity or explicit user request?
    Alignment with User Directives: How well do the assistant's responses match the user's specified needs and expectations?  
    Necessity of Deviations: Are any deviations from instructions made only in situations deemed absolutely necessary or upon  direct user request?  
"""

CODE_EXPLANATION = """
Evaluate the clarity and depth of explanations accompanying code segments. Assess how well the explanation helps in  understanding the code's purpose, logic, and design choices.  

Evaluation Criteria:  
    Clarity: How easy is it to understand the explanation?
    Depth: Does the explanation cover the logic, structure, and decisions behind the code?
    Relevance: Is the explanation relevant to the code's purpose and design philosophy?
    Accessibility: Can a broad audience understand the explanation, regardless of their technical background?  
"""

CODE_COMPLEXITY = """
Evaluate the solutions and code provided by the assistant for their time efficiency and resource management. Assess how  well the code optimizes computational time and resources while ensuring the accuracy and effectiveness of the implemented  algorithms.  

Evaluation Criteria:  
    Time Efficiency: Does the code minimize computational time?  
    Resource Efficiency: Does the code use resources (e.g., memory, CPU) judiciously?  
    Algorithm Effectiveness: Are the chosen algorithms accurate and efficient in achieving the desired outcomes?  
    Optimization: Has the code been optimized for quick processing without compromising the solution's correctness or  efficiency?
"""

CODE_READABILITY = """
Evaluate the readability of code segments. Assess how comments and documentation contribute to understanding the code's  logic, purpose, and operation.  

Evaluation Criteria:  
    Clarity: How clear and understandable are the code and its accompanying comments/documentation?
    Conciseness: Are the comments and documentation succinct yet informative?
    Relevance: Do the comments and documentation directly contribute to explaining the code's logic, objectives, and functionality?
    Comprehensibility: Can users of varying technical backgrounds easily grasp the code's purpose and how it works?  
"""

CODING_STYLE = """
Evaluate the coding style of provided code segments. Assess how well the code adheres to the best practices of the language,  focusing on readability, maintainability, and efficiency in line with the language's idiomatic style.  Evaluation Criteria:  
    Readability: Is the code easy to read and understand?  
    Maintainability: Can the code be easily modified or extended?
    Efficiency: Does the code execute tasks in an efficient manner?  
    Adherence to Idiomatic Style: Does the code follow the stylistic norms and conventions specific to the programming  language? 
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


def clean_llm_output(output: str) -> dict[str, str]:
    output = re.sub(r"^.*?</think>", "", output, flags=re.DOTALL)
    output = output.strip()
    match = re.search(r"```json\n(.*?)\n```", output, re.DOTALL)
    try:
        match = ast.literal_eval(match.group(1).strip())
    except Exception:
        match = {}
    return match


def fill_prompt(preference: str, instruction: str, code1: str, code2: str) -> tuple[str, int]:
    if preference == "complexity":
        evaluation_criteria = CODE_COMPLEXITY
    elif preference == "readability":
        evaluation_criteria = CODE_READABILITY
    elif preference == "style":
        evaluation_criteria = CODING_STYLE
    elif preference == "explanation":
        evaluation_criteria = CODE_EXPLANATION
    elif preference == "instruction-following":
        evaluation_criteria = INSTRUCTION_FOLLOWING
    if random.random() < 0.5:
        # Randomly swap code1 and code2
        return PROMPT_TEMPLATE.format(user_instruction=instruction, code1=code1, code2=code2, evaluation_criteria=evaluation_criteria), 0
    else:
        return PROMPT_TEMPLATE.format(user_instruction=instruction, code1=code2, code2=code1, evaluation_criteria=evaluation_criteria), 1


@hydra.main(version_base=None, config_name="cuf_success_rate_config")
def main(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg.db))
    dataset = datasets.load_dataset(cfg.db.data.name, split=cfg.db.data.split)
    dataset = dataset.shuffle(seed=cfg.db.data.seed).select(range(cfg.db.data.num_samples))
    log.info(f"Loaded {len(dataset)} samples from {cfg.db.data.name}.")
    log.info(f"Using model: {cfg.db.model.name}")
    log.info(f"Using sampling params: {cfg.db.sparams}")
    log.info(f"Preference distribution: {Counter(dataset['preference'])}")

    fill_results = [fill_prompt(x["preference"], x["instruction"], x["chosen"], x["rejected"]) for x in dataset]
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
    gen_answers = [response.get("preferred_code", None) for response in responses]
    correct_answers = ["[[B]]" if swapped else "[[A]]" for swapped in dataset["swapped"]]
    correct = [verdict == answer for verdict, answer in zip(gen_answers, correct_answers)]
    accuracy = sum(correct) / len(correct)
    log.info(f"Model: {cfg.db.model.name}")
    log.info(f"Accuracy: {accuracy:.2%}")
    log.info(f"Correct answers: {sum(correct)}")
    log.info(f"Total answers: {len(correct)}")
    log.info(f"Number of swaps: {sum(dataset['swapped'])}")
    log.info("Completed successfully.")


if __name__ == "__main__":
    main()
