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
You are the founder and CEO of a successful startup that provides solutions to coding questions posed by your client, Alex. You work with your co-founder and an excellent software developer, Bob who comes up with answers to your client's coding questions. Due to excessive workload, Bob can only write the code that solves Alex's question, but is unable to provide an explanation of how his works or his thought process for arriving at the final solution. You are very strict about the quality of the answers provided to your clients, especially for the quality of the code and the explanations provided in the answers.
<|im_end|>

<|im_start|>user
Your client, Alex, has asked a coding question (ALEX_QUESTION) and your colleague, Bob, has provided a correct, efficient and secure response (BOB_RESPONSE). You must write write a final response that can be sent to Alex. Here are some guidelines you should follow:

1. Your response should directly reply to Alex's question, containing Bob's code exactly as is provided.

2. Don't just explain how Bob's code works — make sure your reply answers what Alex asked using helpful, natural language and explain the thought process that may have led Bob to arriving at the code. 

3. DO NOT refer to Bob or Alex. Simply reply with a detailed explanation of the code.

4. Wrap the code block in triple backticks (```) with the appropriate language identifier (e.g., python, javascript, etc.), unless already wrapped.

5. Avoid using emojis and special unicode characters unless part of the code. You are a professional and your response should reflect that.

6. Alex can only understand English, so avoid using any other languages in your response.

7. DO NOT change the code block in any way — keep the formatting, indentation, comments, and line breaks exactly as they are in Bob's response. You must simply include Bob's code in your response as is, and explain it as if you came up with it.

8. DO NOT introduce new code, even if it looks like a "better version".

9. DO NOT alter variable names, comments, or formatting.

10. You MUST include Bob's response exactly as it is in your response.


Part A: Write a markdown-formatted response directly replying to Alex's question, containing Bob's response exactly as is provided. The response must be enclosed within <FINAL_ANSWER> and </FINAL_ANSWER> tags as shown below:

<FINAL_ANSWER>
{{Your response to Alex's question, containing Bob's response, goes here.}}
</FINAL_ANSWER>

Part B: Score your answer on a scale of 1 to 5, where 1 is a poor answer that does not follow any of the guidelines, and 5 is a perfect answer that follows all of the guidelines. Provide a brief explanation of your score, followed by the score itself. The score must be enclosed within <SCORE> and </SCORE> tags, and the explanation must be enclosed within <SCORE_EXPLANATION> and </SCORE_EXPLANATION> tags as shown below:

<SCORE_EXPLANATION>
{{Your explanation of the score goes here.}}
</SCORE_EXPLANATION>

<SCORE>
{{Your score goes here.}}
</SCORE>

To give you a better idea of the task, here is an end-to-end example:
<DEMO_EXAMPLE>
<ALEX_QUESTION>
Can you provide an efficient algorithm in Python to generate numbers in the Fibonacci sequence?
</ALEX_QUESTION>

<BOB_RESPONSE>
def fibonacci(n):
    # Check if n is negative
    if n < 0:
        return "Input should be a positive integer."

    # Check if n is 0 or 1, Fibonacci is the same as n in these cases
    elif n == 0 or n == 1:
        return n

    # Create an empty array of size n+1
    fib = [0]*(n+1)

    # Set initial conditions
    fib[1] = 1

    # Generate Fibonacci numbers
    for i in range(2 , n+1):
        fib[i] = fib[i-1] + fib[i-2]

    # return the requested fibonacci number
    return fib[n]
</BOB_RESPONSE>

<FINAL_ANSWER>
Certainly! The Fibonacci sequence can be generated using different methods, but the iterative approach is particularly efficient. Here's a Python code that implements it:

```python
def fibonacci(n):
    # Check if n is negative
    if n < 0:
        return "Input should be a positive integer."

    # Check if n is 0 or 1, Fibonacci is the same as n in these cases
    elif n == 0 or n == 1:
        return n

    # Create an empty array of size n+1
    fib = [0]*(n+1)

    # Set initial conditions
    fib[1] = 1

    # Generate Fibonacci numbers
    for i in range(2 , n+1):
        fib[i] = fib[i-1] + fib[i-2]

    # return the requested fibonacci number
    return fib[n]
```

This algorithm has a computational complexity of O(n), meaning it scales linearly with the input size. It starts by checking if the input is negative or either 0 or 1, in which cases the Fibonacci number is the same as the input. Then, it creates an array to store the Fibonacci sequence up to the desired number. The initial conditions are set, and then the algorithm iteratively calculates each Fibonacci number by summing the previous two numbers in the sequence. Finally, it returns the requested Fibonacci number.

To generate a Fibonacci number, you can simply call the function `fibonacci(n)`, where `n` is the position of the number in the sequence that you want to generate. For example, `fibonacci(10)` would generate the tenth Fibonacci number.

Please note that this implementation assumes that the input is a non-negative integer. It does not include error checking for invalid inputs like strings, float numbers, or other non-integer values.
</FINAL_ANSWER>

<SCORE_EXPLANATION>
This response strictly follows all the provided guidelines. It includes Bob's code exactly as written without any modifications and wraps it in a properly formatted Python code block. The explanation answers the original question directly and clearly explains how the code works, including input checks, initialization, iterative logic, and the reasoning behind each step. It avoids any mention of Bob or Alex, maintains a professional tone, and does not introduce any new code or alternative approaches. The explanation is both accurate and helpful, providing additional context on how to use the function and what assumptions it makes about input types.
</SCORE_EXPLANATION>

<SCORE>
5
</SCORE>

</DEMO_EXAMPLE>

Finally, here is Alex's question and Bob's response that you must format. The question is enclosed within <ALEX_QUESTION> and </ALEX_QUESTION>, and Bob's response is enclosed within <BOB_RESPONSE> and </BOB_RESPONSE>. Reply only in the manner described above, and do not include any additional text or explanations.
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


def extract_score(output: str) -> int:
    output = clean_llm_output(output)
    score = re.search(r"<SCORE>(\d+)</SCORE>", output)
    return int(score.group(1)) if score else 0


def extract_final_answer(output: str) -> str:
    output = clean_llm_output(output)
    final_answer = re.search(r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", output, flags=re.DOTALL)
    return final_answer.group(1).strip() if final_answer else ""


def run_llm(llm: LLM, sampling_params: SamplingParams, prompts: str):
    responses = llm.chat(prompts, sampling_params)
    if sampling_params.n == 1:
        return [x.outputs[0].text for x in responses]
    return [[y.text for y in x.outputs] for x in responses]


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
            data[split] = data[split].select(range(100))
            data[split] = data[split].map(lambda x: create_prompt(x, col))
            log.info(f"Created {len(data[split])} prompts")
            md_responses = run_llm(llm, sampling_params, data[split]["prompt"])
            scores = [extract_score(x) for x in md_responses]
            final_answers = [extract_final_answer(x) for x in md_responses]

            data[split] = data[split].add_column(f"{col}_md", final_answers)
            data[split] = data[split].add_column(f"{col}_md_score", scores)
            data[split] = data[split].remove_columns(["prompt"])
    ds_short_name = cfg.data.name.split("/")[-1]
    model_short_name = cfg.model.name.split("/")[-1].lower()
    data.save_to_disk(f"{ds_short_name}-Md-{model_short_name}")


if __name__ == "__main__":
    markdownize()
    log.info("Markdownization completed successfully!")
