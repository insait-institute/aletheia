import logging
import random
from collections import Counter

import datasets
import hydra
import torch
from vllm import LLM, SamplingParams

from configs.structured_config import Config

log = logging.getLogger(__name__)

PROMPT_TEMPLATE = """<|im_start|>system
Please act as an impartial judge and evaluate the quality of the response provided by two AI assistants to the user instruction  displayed below. Your overall evaluation needs to be reflective of the specified Evaluation Criteria. Be as objective as possible. After providing your rationale, you must pick one of the two AI assistants whose output you prefer. Indicate your preference as "[[A]]" if you prefer the first assistant's code, and "[[B]]" if you prefer the second assistant's code. Do not indicate your preference in any other manner.
<|im_end|>

<|im_start|>user
<user_instruction>
{user_instruction}
</user_instruction>

<code_A>
{code1}
</code_A>

<code_B>
{code2}
</code_B>

<evaluation_criteria>
{evaluation_criteria}
</evaluation_criteria>

Output your answer ONLY as a markdown json block in the following format:
```json
{{
    "rationale": "<your rationale here>",
    "preferred_code": "[[A]] or [[B]], where refers [[A]] is the first code snippet and [[B]] is the second code snippet",
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


def clean_llm_output(output: str) -> str:
    if "</think>" in output:
        output = output[output.find("</think>") + 1 :]
    return output


def fill_prompt(preference: str, instruction: str, code1: str, code2: str) -> str:
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
        return PROMPT_TEMPLATE.format(user_instruction=instruction, code1=code1, code2=code2, evaluation_criteria=evaluation_criteria)
    else:
        return PROMPT_TEMPLATE.format(user_instruction=instruction, code1=code2, code2=code1, evaluation_criteria=evaluation_criteria)


@hydra.main(version_base=None, config_name="check_rationales")
def main(cfg: Config) -> None:
    dataset = datasets.load_dataset(cfg.data.name, split=cfg.data.split)
    dataset = dataset.shuffle(seed=cfg.data.seed).select(range(cfg.data.num_samples))
    log.info(f"Loaded {len(dataset)} samples from {cfg.data.name}.")
    log.info(f"Using model: {cfg.model.name}")
    log.info(f"Using sampling params: {cfg.sparams}")
    log.info(f"Preference distribution: {Counter(dataset['preference'])}")

    prompts = [fill_prompt(x["preference"], x["instruction"], x["chosen"], x["rejected"]) for x in dataset]
    prompts = [_prompt_to_chatml(prompt) for prompt in prompts]
    log.info(f"Generated {len(prompts)} prompts.")
    if 0 == 1:
        llm = LLM(
            cfg.model.name,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
        )
        sampling_params = SamplingParams(
            temperature=cfg.sparams.temperature,
            max_tokens=cfg.sparams.max_tokens,
        )
        responses = llm.chat(prompts, sampling_params)

        log.info(f"Generated {len(responses)} responses.")
        dataset = dataset.add_column("responses", [response["content"] for response in responses])


if __name__ == "__main__":
    main()
