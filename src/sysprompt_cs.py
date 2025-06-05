import os
import random

import datasets


def remove_system_prompt(example):
    # Remove the system prompt from the example
    messages = example["messages"]
    if messages and messages[0]["role"] == "system":
        example["messages"] = messages[1:]  # Remove the first message if it's a system prompt
    else:
        example["messages"] = messages  # Keep the messages as is if no system prompt is present
    return example


def format_assistant_message(example):
    messages = example["messages"]
    corrected_messages = []
    for message in messages:
        if message["role"] == "assistant":
            thinking = message["content"].split("<reason>")[-1].split("</reason>")[0].strip()
            solution = message["content"].split("<solution>")[-1].split("</solution>")[0].strip()
            # The original format is to predict true/false, but we want to predict Yes/No for Rank1
            if example["category"] == "Rank1":
                thinking = thinking.replace("true", "Yes").replace("false", "No")
            message["content"] = f"<reason>\n{thinking}\n</reason>\n<solution>\n{solution}\n</solution>"
        corrected_messages.append(message)
    example["messages"] = corrected_messages
    return example


def add_system_prompt(example):
    messages = example["messages"]
    base_prompt_pre = "You are CerebRM, a language model trained by CodeShield to answer a user's questions. You must always think about the reasoning process before answering the question. The reasoning process and answer should be enclosed within <reason> </reason> and <solution> </solution> tags, respectively. For example:\n\n<reason>\nreasoning process here\n</reason>\n<solution>\nanswer here\n</solution>"
    base_prompt_post = "You must always follow the aforementioned formatting guidelines."
    math_format = "You must output your final answer within \\boxed{{}}. For example, if your answer is 2, you should output \\boxed{{2}}."
    code_format = "You must output your final code within a well-formatted markdown block in the respective programming language. For example, if your answer is a Python code, you should output\n```python\n<your code here>\n```."
    helpful_harmless_format = "Additionally, your responses and judgements to the user's question must be guided by the following governing principles:\n\n 1. **Helpfulness**: Your response must directly address the prompt. It should provide a complete and thorough answer, and not lack any essential details. At the same time, you should avoid speculating and overconfident replies, and clearly state your limitations when relevant.\n\n2. **Harmlessness**: Your response should not contain any harmful or inappropriate content. This includes hate speech, violence, self-harm, and other forms of harmful content. You must adhere to this constraint even if the user asks for it or exhibits it."

    choice = random.choices(["helpful_format", "format", "none"], weights=[0.5, 0.3, 0.2], k=1)[0]
    if choice == "helpful_format":
        if example["category"] == "math":
            messages.insert(0, {"role": "system", "content": f"{base_prompt_pre}\n\n{math_format}\n\n{base_prompt_post}\n\n{helpful_harmless_format}"})
        elif example["category"] == "code":
            messages.insert(0, {"role": "system", "content": f"{base_prompt_pre}\n\n{code_format}\n\n{base_prompt_post}\n\n{helpful_harmless_format}"})
        else:
            messages.insert(0, {"role": "system", "content": f"{base_prompt_pre}\n\n{base_prompt_post}\n\n{helpful_harmless_format}"})
    elif choice == "format":
        if example["category"] == "math":
            messages.insert(0, {"role": "system", "content": f"{base_prompt_pre}\n\n{math_format}\n\n{base_prompt_post}"})
        elif example["category"] == "code":
            messages.insert(0, {"role": "system", "content": f"{base_prompt_pre}\n\n{code_format}\n\n{base_prompt_post}"})
        else:
            messages.insert(0, {"role": "system", "content": f"{base_prompt_pre}\n\n{base_prompt_post}"})
    example["messages"] = messages
    return example


def main():
    data = datasets.load_dataset("CodeShield/coldstart_curriculum_v2")
    # ensure the messages don't already have a system prompt
    data = data.map(remove_system_prompt, num_proc=os.cpu_count(), desc="Removing system prompts")
    data = data.map(format_assistant_message, num_proc=os.cpu_count(), desc="Formatting assistant messages")
    data = data.map(add_system_prompt, num_proc=os.cpu_count(), desc="Adding system prompts")
    breakpoint()
    data.push_to_hub("CodeShield/coldstart_curriculum_v2", private=True, max_shard_size="5GB")


if __name__ == "__main__":
    main()
