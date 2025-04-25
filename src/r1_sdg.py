import asyncio
import logging
import os
import pickle
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import backoff
import datasets
import hydra
import openai
from openai import ChatCompletion
from tqdm.asyncio import tqdm

from configs.r1_sdg_config import Config

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENROUTER_API_BASE")
client = openai.AsyncOpenAI(api_key=api_key, base_url=api_base)

REWARD_PROMPT_TEMPLATE = """<|im_start|>system
You are an excellent and professional judge. You are given two responses to the same question, one of which is of a higher quality than the other. Your task is to assign a numeric score to each response based on the following criteria:
1. **Correctness**: Is the response correct? Does it answer the user's question completely and accurately?
2. **Clarity**: How clear and understandable are the code and the accompanying comments and documentation?
3. **Efficiency**: Is the code efficient? Does it follow best practices for performance and resource usage? Does it avoid unnecessary complexity?
4. **Conciseness**: Is the response concise and to the point? Does it avoid unnecessary repetition or verbosity?
5. **Security**: Does the response follow best practices for security? Does it avoid common pitfalls and vulnerabilities?
6. **Style**: Does the code follow the stylistic norms and idioms specific to the programming language? Is the code well-structured and easy to read?

You will be told which response is of a higher quality, and you must assign a score to both responses based on the above criteria.
<|im_end|> 

<|im_start|>user
Here are the user's question (within <USER_QUESTION> and </USER_QUESTION>), the higher quality response (within <GOOD_RESPONSE> and </GOOD_RESPONSE>), and the lower quality response (within <POOR_RESPONSE> and </POOR_RESPONSE>).
<USER_QUESTION>
{query}
</USER_QUESTION>

<GOOD_RESPONSE>
{response_1}
</GOOD_RESPONSE>

<POOR_RESPONSE>
{response_2}
</POOR_RESPONSE>

Reply with a brief analysis of the two responses, comparing them based on the criteria provided. Then, assign a score to each response between 1 and 10, where 1 is the lowest quality and 10 is the highest quality.
Naturally, the higher quality response should receive a higher score than the lower quality response. The difference between the two scores must accurately and consistently reflect the difference in quality between the two responses, with a large difference in score implying a large difference in response quality, and vice-versa.
Your response must exactly follow the format below:
<ANALYSIS>
{{Compare different responses based on the given criteria}}
</ANALYSIS>

<SCORES>
{{the overall comprehensive score of both responses in order, separate by comma in the format \\boxed{{x, y}}, where x and y are the scores for response 1 and response 2 respectively.}}
</SCORES>
<|im_end|>
"""


def _prompt_to_chatml(prompt: str, start_token: str = "<|im_start|>", end_token: str = "<|im_end|>") -> Dict[str, str]:
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


def _string_to_dict(to_convert: str) -> Dict[str, str]:
    return {s.split("=", 1)[0]: s.split("=", 1)[1] for s in to_convert.split(" ") if len(s) > 0}


def create_prompt(example: Dict[str, Any], input_col: str, chosen_col: str, rejected_col: str) -> List[Dict[str, str]]:
    example["prompt"] = REWARD_PROMPT_TEMPLATE.format(query=example[input_col], response_1=example[chosen_col], response_2=example[rejected_col])
    example["prompt"] = "<|im_start|>user\n who are you?<|im_end|"
    example["prompt"] = _prompt_to_chatml(example["prompt"])
    return example


def _get_message(response: ChatCompletion) -> str:
    return response.choices[0].message


def get_reasoning_trace(response: ChatCompletion) -> str:
    return _get_message(response).reasoning


def get_analysis(response: ChatCompletion) -> str:
    message = _get_message(response).content
    match = re.match(r"<ANALYSIS>\s*(.*?)\s*</ANALYSIS>", message, flags=re.DOTALL)
    return match.group(1).strip() if match else None


def get_scores(response: ChatCompletion) -> Tuple[int, int]:
    message = _get_message(response).content
    try:
        # Extract content between <SCORES> tags
        match = re.search(r"<SCORES>\s*(.*?)\s*</SCORES>", message, flags=re.DOTALL)
        if not match:
            raise ValueError("No <SCORES> block found")

        scores_block = match.group(1)

        # Extract numbers from \boxed{8,2} or \boxed{{8,2}} formats
        boxed_match = re.search(r"\\boxed(?:\{+)([\d\s,]+)(?:\}+)", scores_block)
        if not boxed_match:
            raise ValueError("No \\boxed{{}} content found")

        scores = boxed_match.group(1).strip().split(",")
        assert len(scores) == 2, "Expected two scores"
        return int(scores[0]), int(scores[1])
    except Exception:
        log.error(f"Error parsing scores: {traceback.format_exc()}")
        return None, None


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.BadRequestError, openai.APIConnectionError, openai.APITimeoutError))
async def _get_single_response(prompt: List[Dict[str, str]], sparams: Dict[str, Any]) -> ChatCompletion:
    response = await client.chat.completions.create(
        messages=prompt,
        model="deepseek/deepseek-r1",
        max_tokens=sparams.max_tokens,
        seed=sparams.seed,
        extra_body={
            "provider": {"order": ["DeepInfra", "Lambda"], "require_parameters": True},
        },
    )
    return response


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


async def get_responses(prompts: List[List[Dict[str, str]]], sparams: Dict[str, Any]) -> List[ChatCompletion]:
    results = []
    save_dir = Path(__file__).parent.parent / "outputs/sdg"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Find last saved chunk
    last_processed_chunk = -1
    intermediate_files = list(save_dir.glob("intermediate-sdg-output-*.pkl"))
    if intermediate_files:
        # If a file exists, extract the chunk number from its name
        file = intermediate_files[0]
        match = re.search(r"intermediate-sdg-output-(\d+).pkl", file.name)
        if match:
            last_processed_chunk = int(match.group(1))

    # Load intermediate results if any
    if last_processed_chunk != -1:
        with open(save_dir / f"intermediate-sdg-output-{last_processed_chunk}.pkl", "rb") as f:
            results = pickle.load(f)
            log.info(f"Loaded intermediate results from batch {last_processed_chunk}")

    # Iterate chunk by chunk
    for batch_num, batch in enumerate(chunks(prompts, sparams.chunk_batch_size), start=1):
        # Skip chunk if processed
        if batch_num <= last_processed_chunk:
            log.info(f"Skipping batch {batch_num} (already processed)")
            continue
        log.info(f"Processing batch {batch_num} ({len(batch)} prompts)...")
        sem = asyncio.Semaphore(sparams.chunk_max_concurrency)

        async def sem_fetch(p):
            async with sem:
                return await _get_single_response(p, sparams)

        coros = [sem_fetch(p) for p in batch]
        batch_responses = await tqdm.gather(
            *coros,
            total=len(coros),
            desc=f"Batch {batch_num}",
        )
        results.extend(batch_responses)

        # Save intermediate results
        with open(save_dir / f"intermediate-sdg-output-{batch_num}.pkl", "wb") as f:
            pickle.dump(batch_responses, f)
            # Remove older pickle files (keep only the current batch file)
            for old_file in save_dir.glob("intermediate-sdg-output-*.pkl"):
                if old_file.name != f"intermediate-sdg-output-{batch_num}.pkl":
                    old_file.unlink()
            log.info(f"Batch {batch_num} responses saved")
    return results


@hydra.main(version_base=None, config_name="r1_sdg_config")
def main(cfg: Config) -> None:
    data = datasets.load_dataset(cfg.data.name)
    train_data = data[cfg.data.train_split]
    test_data = data[cfg.data.test_split]

    train_data = train_data.map(lambda x: create_prompt(x, cfg.data.input_col, cfg.data.chosen_col, cfg.data.rejected_col))
    test_data = test_data.map(lambda x: create_prompt(x, cfg.data.input_col, cfg.data.chosen_col, cfg.data.rejected_col))

    all_prompts = train_data["prompt"] + test_data["prompt"]
    all_responses = asyncio.run(get_responses(all_prompts, sparams=cfg.sparams))

    reasoning_traces = [get_reasoning_trace(response) for response in all_responses]
    analysis = [get_analysis(response) for response in all_responses]
    score_tuples = [get_scores(response) for response in all_responses]

    train_data = train_data.add_column("reasoning_trace", reasoning_traces[: len(train_data)])
    train_data = train_data.add_column("analysis", analysis[: len(train_data)])
    train_data = train_data.add_column("score_chosen", [score[0] for score in score_tuples[: len(train_data)]])
    train_data = train_data.add_column("score_rejected", [score[1] for score in score_tuples[: len(train_data)]])

    test_data = test_data.add_column("reasoning_trace", reasoning_traces[len(train_data) :])
    test_data = test_data.add_column("analysis", analysis[len(train_data) :])
    test_data = test_data.add_column("score_chosen", [score[0] for score in score_tuples[len(train_data) :]])
    test_data = test_data.add_column("score_rejected", [score[1] for score in score_tuples[len(train_data) :]])

    train_data = train_data.remove_columns(["prompt"])
    test_data = test_data.remove_columns(["prompt"])

    dataset = datasets.DatasetDict(
        {
            "train": train_data,
            "test": test_data,
        }
    )
    dataset.push_to_hub(
        repo_id="wetsoledrysoul/CPE-R1-Rewards",
        private=True,
    )
    log.info("Dataset pushed to Hugging Face Hub")


if __name__ == "__main__":
    main()
