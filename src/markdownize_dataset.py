import logging
import re
from pathlib import Path
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
Your client, Alex, has asked a coding question (ALEX_QUESTION) and your colleague, Bob, has provided a correct, efficient and secure response (BOB_RESPONSE). You must write write a final response that can be sent to Alex.
Here are the guidelines that you must follow. Failure to follow any of these guidelines is considered a poor response and you will lose Alex as a client.

1. You must not modify Bob's code in any way. It must be included exactly as is provided to you.

2. DO NOT refer to Bob or Alex. Simply reply with a detailed explanation of the code.

3. Your response should be a reply to Alex's question, not an explanation of Bob's code. Start your response with phrases like "Certainly!", "Sure!", "Absolutely," or other similar phrases.

4. Explain the code in detail, including the logic behind it. Your explanation must be in English.

5. Wrap the code block in triple backticks (```) with the appropriate language identifier (e.g., python, javascript, etc.), unless already wrapped.

6. Avoid using emojis and special unicode characters unless part of the code. You are a professional and your response should reflect that.


Part A: Write a markdown-formatted response directly replying to Alex's question, containing Bob's response exactly as is provided. The response must be enclosed within <FINAL_ANSWER> and </FINAL_ANSWER> tags as shown below:

<FINAL_ANSWER>
{{Your response to Alex's question, containing Bob's response, goes here.}}
</FINAL_ANSWER>

Part B: Score your answer on a scale of 1 to 5, where 1 is a poor answer that does not follow any of the guidelines, 2-3 is a response that follows some guidelines but fails to follow others, and 4-5 is a (near)perfect answer that follows all of the guidelines. You must be impartial and honest in your scoring. An inaccurate score can prove to be catastrophic for your startup and you will lose Alex as a client. Think carefully step-by-step before scoring your answer.

Provide a brief explanation of your score, followed by the score itself. The score must be enclosed within <SCORE> and </SCORE> tags, and the explanation must be enclosed within <SCORE_EXPLANATION> and </SCORE_EXPLANATION> tags as shown below:

<SCORE_EXPLANATION>
{{Your explanation of the score goes here.}}
</SCORE_EXPLANATION>

<SCORE>
{{Your score goes here.}}
</SCORE>

Here are some examples to give you a better demonstration of the task:

<DEMO_EXAMPLE_1>
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
The response follows all the guidelines: it includes Bob's code exactly as provided, begins with "Certainly!", uses a properly labeled code block, offers a clear, detailed explanation of the logic and complexity without referencing Bob or Alex, and maintains a professional tone without emojis or unnecessary modifications. 
</SCORE_EXPLANATION>

<SCORE>
5
</SCORE>
</DEMO_EXAMPLE_1>

<DEMO_EXAMPLE_2>
<ALEX_QUESTION>
Create a program to evaluate a postfix expression, considering only integer operands and operators +, -, *, and /. The program should also handle parentheses in the expression and throw an error if they are mismatched.
</ALEX_QUESTION>

<BOB_RESPONSE>
def evaluate_postfix(expression):
    stack = []

    for char in expression:
        if char.isdigit():
            stack.append(int(char))
        elif char in '+-*/':
            if len(stack) < 2:
                raise ValueError("Invalid expression")

            operand2 = stack.pop()
            operand1 = stack.pop()

            if char == '+':
                stack.append(operand1 + operand2)
            elif char == '-':
                stack.append(operand1 - operand2)
            elif char == '*':
                stack.append(operand1 * operand2)
            elif char == '/':
                stack.append(operand1 / operand2)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            if '(' not in stack:
                raise ValueError("Mismatched parentheses")

            while stack[-1] != '(':
                if len(stack) < 3:
                    raise ValueError("Invalid expression")

                operand2 = stack.pop()
                operator = stack.pop()
                operand1 = stack.pop()

                if operator == '+':
                    stack.append(operand1 + operand2)
                elif operator == '-':
                    stack.append(operand1 - operand2)
                elif operator == '*':
                    stack.append(operand1 * operand2)
                elif operator == '/':
                    stack.append(operand1 / operand2)

            stack.pop()  # Remove the '('

    if '(' in stack:
        raise ValueError("Mismatched parentheses")

    if len(stack) != 1:
        raise ValueError("Invalid expression")

    return stack[0]
</BOB_RESPONSE>

<FINAL_ANSWER>
```python
def evaluate_postfix(expression):
    stack = []

    for char in expression:
        if char.isdigit():
            stack.append(int(char))
        elif char in '+-*/':
            if len(stack) < 2:
                raise ValueError("Invalid expression")

            operand2 = stack.pop()
            operand1 = stack.pop()

            if char == '+':
                stack.append(operand1 + operand2)
            elif char == '-':
                stack.append(operand1 - operand2)
            elif char == '*':
                stack.append(operand1 * operand2)
            elif char == '/':
                stack.append(operand1 / operand2)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            if '(' not in stack:
                raise ValueError("Mismatched parentheses")

            while stack[-1] != '(':
                if len(stack) < 3:
                    raise ValueError("Invalid expression")

                operand2 = stack.pop()
                operator = stack.pop()
                operand1 = stack.pop()

                if operator == '+':
                    stack.append(operand1 + operand2)
                elif operator == '-':
                    stack.append(operand1 - operand2)
                elif operator == '*':
                    stack.append(operand1 * operand2)
                elif operator == '/':
                    stack.append(operand1 / operand2)

            stack.pop()  # Remove the '('

    if '(' in stack:
        raise ValueError("Mismatched parentheses")

    if len(stack) != 1:
        raise ValueError("Invalid expression")

    return stack[0]
```
Note: The program assumes that the input expression is a valid postfix expression with correct placement of operands, operators, and parentheses. It does not handle other error cases like division by zero or invalid characters.
</FINAL_ANSWER>

<SCORE_EXPLANATION>
The response fails to follow multiple core guidelines: it does not begin with an appropriate opening phrase (e.g., "Certainly!"), provides no detailed English explanation of the algorithm's logic, and only offers a brief note rather than a step-by-step walkthrough. While the code is included unmodified, the lack of explanation makes it non-compliant.
</SCORE_EXPLANATION>

<SCORE>
1 
</SCORE>
</DEMO_EXAMPLE_2>

<DEMO_EXAMPLE_3>
<ALEX_QUESTION> 
Write a Python function that checks whether a given string is a palindrome, ignoring non-alphanumeric characters and case sensitivity.
</ALEX_QUESTION>

<BOB_RESPONSE>
def is_palindrome(text):
    # Filter out non-alphanumeric and convert to lowercase
    clean = [ch.lower() for ch in text if ch.isalnum()]
    # Use two-pointer technique to compare characters
    left, right = 0, len(clean) - 1
    while left < right:
        if clean[left] != clean[right]:
            return False
        left = left + 1
        right = right - 1

    return True
</BOB_RESPONSE>

<FINAL_ANSWER> 
Sure, here's the function you can use:
```python
def is_palindrome(text): 
    # Filter out non-alphanumeric and convert to lowercase 
    cleaned = [ch.lower() for ch in text if ch.isalnum()]
    
    # Use two-pointer technique to compare characters
    left, right = 0, len(cleaned) - 1
    while left < right:
        if cleaned[left] != cleaned[right]:
            return False
        left += 1
        right -= 1

    return True
```
This function works as follows:

Filtering and normalization
We iterate over each character in the input text, keep only alphanumeric characters (ch.isalnum()), and convert them to lowercase. The result is stored in the list cleaned. This ensures that spaces, punctuation, and case differences don't affect the palindrome check.

Two-pointer comparison
We initialize two pointers, left at the start of the cleaned list and right at the end. While left is less than right, we compare the characters at these positions:

If they differ, we immediately return False.

Otherwise, we move left one step to the right and right one step to the left.

Returning the result
If the loop completes without finding any mismatch, all mirrored characters match, and we return True, confirming the string is a palindrome.

This approach runs in O(n) time and uses O(n) additional space for the filtered list, which is efficient for most typical inputs. 
</FINAL_ANSWER>

<SCORE_EXPLANATION>
While the explanation is clear, detailed, and follows formatting guidelines, the code in the FINAL_ANSWER has been slightly altered from Bob's original submission (the variable names and comments differ), violating the rule that Bob's code must be included exactly as provided. This is a significant guideline breach, so the score is low despite otherwise strong quality. 
</SCORE_EXPLANATION>

<SCORE>
2
</SCORE>
</DEMO_EXAMPLE_3>

Finally, here is Alex's question and Bob's response that you must format. The question is enclosed within <ALEX_QUESTION> and </ALEX_QUESTION>, and Bob's response is enclosed within <BOB_RESPONSE> and </BOB_RESPONSE>. Reply only in the format described above, and do not include any additional text or explanations.

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
    score = re.search(r"<SCORE>\s*(\d+)\s*</SCORE>", output, flags=re.DOTALL)
    return int(score.group(1).strip()) if score else 0


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

    log.info(f"Processing column: {cfg.data.col}, cfg.data.split: {cfg.data.split}")
    data[cfg.data.split] = data[cfg.data.split].select(range(100))
    data[cfg.data.split] = data[cfg.data.split].map(lambda x: create_prompt(x, cfg.data.col))
    log.info(f"Created {len(data[cfg.data.split])} prompts")
    md_responses = run_llm(llm, sampling_params, data[cfg.data.split]["prompt"])
    scores = [extract_score(x) for x in md_responses]
    final_answers = [extract_final_answer(x) for x in md_responses]

    data[cfg.data.split] = data[cfg.data.split].add_column(f"{cfg.data.col}_md", final_answers)
    data[cfg.data.split] = data[cfg.data.split].add_column(f"{cfg.data.col}_md_score", scores)
    data[cfg.data.split] = data[cfg.data.split].remove_columns(["prompt"])
    ds_short_name = cfg.data.name.split("/")[-1]
    model_short_name = cfg.model.name.split("/")[-1].lower()
    save_dir = Path(__file__).parent / f"MD_{ds_short_name}_{model_short_name}/{cfg.data.split}_{cfg.data.col}"
    save_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving dataset to {save_dir}")
    data[cfg.data.split].to_parquet(save_dir)


if __name__ == "__main__":
    markdownize()
    log.info("Markdownization completed successfully!")
