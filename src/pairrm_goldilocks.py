import logging
import os
import pickle
import random
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from datasets import Dataset, concatenate_datasets, load_dataset
from vllm import LLM, SamplingParams

from configs.goldilocks_config import Config

log = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
log.setLevel(logging.INFO)
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
REASON_SYSPROMPT = """You must always think about the reasoning process before answering the question. The reasoning process and answer should be enclosed within <reason> </reason> and <solution> </solution> tags, respectively. For example:

<reason>
reasoning process here
</reason>
<solution>
answer here
</solution>

You must always adhere to the above format, in addition to any constraints the user may impose.
"""


def shard_prompts(prompts: list, n: int) -> list[list]:
    k, r = divmod(len(prompts), n)
    return [prompts[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


def construct_prompt(example, idx):
    chosen = example["chosen"]
    rejected = example["rejected"]

    example["correct_ans"] = "[[A]]"
    example["idx"] = idx
    example["num_principles"] = 0
    sys_prompt = REASON_SYSPROMPT

    if random.random() < 0.5:
        example["correct_ans"] = "[[B]]"
        chosen, rejected = rejected, chosen

    if example["system"]:
        example["num_principles"] = int(example["system"].strip().split("\n\n")[-1][0])
        sys_prompt = example["system"].strip() + "\n\n" + REASON_SYSPROMPT

    example["prompt"] = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"You will be shown a question and two responses, A and B. You must carefully evaluate both the responses and indicate which one is better according to the evaluation criteria provided.\n\n[QUESTION]\n{example['input']}\n[/QUESTION]\n\n[RESPONSE_A]\n{chosen}\n[/RESPONSE_A]\n\n[RESPONSE_B]\n{rejected}\n[/RESPONSE_B]\nYour final answer should be '[[A]]' if you think Response A is better with respect to the evaluation criteria, and '[[B]]' if you think Response B is better. Any other response will be immediately rejected. Only reply with '[[A]]' or '[[B]]'.",
        },
        {"role": "assistant", "content": "<reason>"},
    ]
    return example


def filter_by_source(example, source, membership):
    if membership:
        return example["source"] == source
    return example["source"] != source


def get_reasoning(response):
    reasoning = response.split("<reason>")[-1].split("</reason>")[0]
    return reasoning.strip()


def get_answer(response):
    answer = response.split("<solution>")[-1].split("</solution>")[0].strip()
    return answer.strip()


def score_answers(answers, correct_answer, strict=True):
    if strict:
        return [int(x == correct_answer) for x in answers]
    else:
        wrong_answer = "[[B]]" if correct_answer == "[[A]]" else "[[A]]"
        return [int(correct_answer in x and wrong_answer not in x) for x in answers]


@hydra.main(version_base=None, config_name="goldilocks_config")
def main(cfg: Config) -> None:
    assert cfg.inference.goldilocks_type in ["all", "commitpref"], "goldilocks_type must be either 'all' or 'commitpref'"
    if cfg.inference.goldilocks_type == "all":
        general_preference = load_dataset("CodeShield/General-Preference")["train"]
        general_preference = general_preference.add_column("subset", ["genpref"] * len(general_preference))
        commit_preference_enhanced = load_dataset("CodeShield/Commit-Preference-Enhanced")["train"]
        commit_preference: Dataset = commit_preference_enhanced.filter(
            filter_by_source,
            fn_kwargs={"source": "COMMIT_PREFS", "membership": True},
            num_proc=NUM_WORKERS,
            desc="Constructing commit preference dataset",
        )
        enhanced: Dataset = commit_preference_enhanced.filter(
            filter_by_source,
            fn_kwargs={"source": "COMMIT_PREFS", "membership": False},
            num_proc=NUM_WORKERS,
            desc="Constructing enhanced dataset",
        )

        commit_preference = commit_preference.add_column("subset", ["commitpref"] * len(commit_preference))
        enhanced = enhanced.add_column("subset", ["enhanced"] * len(enhanced))

        data: Dataset = concatenate_datasets(
            [
                general_preference,
                commit_preference.shuffle(42).select(range(50000)),
                enhanced.shuffle(42).select(range(70000)),
            ]
        )
    else:
        commit_preference_enhanced = load_dataset("CodeShield/Commit-Preference-Enhanced")["train"]
        data: Dataset = commit_preference_enhanced.filter(
            filter_by_source,
            fn_kwargs={"source": "COMMIT_PREFS", "membership": True},
            num_proc=NUM_WORKERS,
            desc="Constructing commit preference dataset",
        )

    data = data.map(
        construct_prompt,
        num_proc=NUM_WORKERS,
        desc="Constructing prompts",
        with_indices=True,
        remove_columns=["chosen", "rejected", "input", "system"],
    )
    llm = LLM(
        model=cfg.inference.model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=8192,
        n=cfg.inference.N,
        seed=42,
        skip_special_tokens=False,
    )
    sharded_idx = shard_prompts(range(len(data)), cfg.inference.total_shards)[cfg.inference.shard]

    data = data.select(sharded_idx).flatten_indices()
    prompts = data["prompt"]

    log.info(f"Running inference on shard {cfg.inference.shard + 1}/{cfg.inference.total_shards}: {len(prompts)} prompts")
    responses = llm.chat(prompts, sampling_params, continue_final_message=True, add_generation_prompt=False)
    response_texts = [[nth.text for nth in response.outputs] for response in responses]
    BASE_DIR = Path(__file__).parent.parent / cfg.inference.output_dir / cfg.inference.goldilocks_type
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    model_short_name = cfg.inference.model_name.split("/")[-1].split("-")[-1]
    responses_outfile = BASE_DIR / f"reasoning_traces_shard_{cfg.inference.shard}_{model_short_name}.pkl"
    with open(responses_outfile, "wb") as f:
        pickle.dump({i: traces for i, traces in enumerate(response_texts)}, f)
    final_answers = [[get_answer(text) for text in response] for response in response_texts]
    correctness_strict = [score_answers(ans, c_ans) for ans, c_ans in zip(final_answers, data["correct_ans"])]
    group_accuracy_strict = [float(np.mean(corr)) for corr in correctness_strict]

    correctness_relaxed = [score_answers(ans, c_ans, strict=False) for ans, c_ans in zip(final_answers, data["correct_ans"])]
    group_accuracy_relaxed = [float(np.mean(corr)) for corr in correctness_relaxed]

    data = data.add_column("strict_accuracy", group_accuracy_strict)
    data = data.add_column("relaxed_accuracy", group_accuracy_relaxed)
    data = data.add_column("final_answers", [str(ans) for ans in final_answers])

    df = pl.from_arrow(data.data.table)
    df.write_excel(BASE_DIR / f"evaluation_results_shard_{cfg.inference.shard}_{model_short_name}.xlsx")


if __name__ == "__main__":
    main()
