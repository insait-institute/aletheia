import argparse
import logging
import os
import random

from datasets import load_dataset

from cerebrm_prompts import LIST_REWARD_PROMPT
from cerebrm_rewards import extract_boxed_contents_list
from utils import run_inference

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_prompts(example):
    example["correct_ans"] = "A"
    candidate_str = f"[CANDIDATE_A]\n```{example['language']}\n{example['chosen']}\n```\n[/CANDIDATE_A]\n\n[CANDIDATE_B]\n```{example['language']}\n{example['rejected']}\n```\n[/CANDIDATE_B]"
    if random.random() > 0.5:
        example["correct_ans"] = "B"
        candidate_str = f"[CANDIDATE_A]\n```{example['language']}\n{example['rejected']}\n```\n[/CANDIDATE_A]\n\n[CANDIDATE_B]\n```{example['language']}\n{example['chosen']}\n```\n[/CANDIDATE_B]"

    PROMPT = LIST_REWARD_PROMPT
    valid_options = "A, B"

    example["prompt"] = [
        {
            "role": "user",
            "content": PROMPT.format(
                question=example["query"],
                candidates=candidate_str,
                valid_options=valid_options,
            ).strip(),
        },
    ]
    return example


def check_ans_switch(example, idx_to_correct_pre):
    post_correct = example["model_correct"]
    post_idx = example["idx"]
    pre_correct = idx_to_correct_pre[post_idx]
    example["pre_correct"] = pre_correct
    if pre_correct and not post_correct:
        example["ans_switch"] = "switched_to_incorrect"
    elif not pre_correct and post_correct:
        example["ans_switch"] = "switched_to_correct"
    else:
        example["ans_switch"] = "no_change"
    return example


def main(args):
    data = load_dataset("wetsoledrysoul/RQ4-Set")
    data_pre = data["original"]
    data_post = data[args.split]

    data_pre = data_pre.map(_create_prompts, num_proc=NUM_WORKERS, desc="Creating prompts")
    data_post = data_post.map(_create_prompts, num_proc=NUM_WORKERS, desc="Creating prompts")

    prompts = list(data_pre["prompt"]) + list(data_post["prompt"])
    completions = run_inference(
        prompts,
        args.eval_llm,
        temperature=0.6,
        max_tokens=args.max_tokens,
        max_model_len=args.max_tokens + 4096,
        tp_size=1,
        top_p=0.95,
        n=1,
        gpu_memory_utilization=0.95,
    )
    completions = [response.outputs[0].text for response in completions]
    model_answers = [extract_boxed_contents_list(x) for x in completions]
    expected_answers = list(data_pre["correct_ans"]) + list(data_post["correct_ans"])

    correct_or_not = []
    for correct_ans, answer in zip(expected_answers, model_answers):
        correct_or_not.append(answer == correct_ans)

    pre_correct = correct_or_not[: len(data_pre)]
    post_correct = correct_or_not[len(data_pre) :]
    data_pre = data_pre.add_column("model_correct", pre_correct)
    data_post = data_post.add_column("model_correct", post_correct)

    idx_to_correct_pre = {data_pre[i]["idx"]: data_pre[i]["model_correct"] for i in range(len(data_pre))}

    data_post = data_post.map(check_ans_switch, fn_kwargs={"idx_to_correct_pre": idx_to_correct_pre}, num_proc=NUM_WORKERS, desc="Checking answer switches")
    data_post = data_post.to_pandas()
    data_post.to_csv(f"outputs/rq4_validation_{args.split}.csv", index=False)
    log.info("---- Evaluation Results ----")
    log.info(f"{data_post[['modification', 'ans_switch']].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["chosen_worsened", "rejected_enhanced"], required=True, help="Dataset to use for evaluation")
    parser.add_argument("--eval_llm", type=str, default="deepseek-ai/Deepseek-R1-Distill-Qwen-7B", help="LLM to use for evaluation")
    parser.add_argument("--K", type=int, default=1, help="Number of samples to generate for each prompt")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum number of tokens to generate")
    args = parser.parse_args()
    main(args)
