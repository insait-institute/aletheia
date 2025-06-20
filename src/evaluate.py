import logging
import os
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from configs.evaluate_config import Config

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

N = 8
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()


def last_index(lst, value):
    return len(lst) - lst[::-1].index(value) - 1


def compute_score(yes_logprob, no_logprob):
    return np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))


def filter_by_origin(example, origin):
    return "_".join(example["instance_id"].split("_")[:-1]) == origin


def compute_final_score(group_df):
    yes_row = group_df.filter(pl.col("verdict") == "Yes")
    no_row = group_df.filter(pl.col("verdict") == "No")

    score = 1 if yes_row["score"][0] > no_row["score"][0] else 0
    return pl.Series(name="final_score", values=[score, score])


@hydra.main(version_base=None, config_name="evaluate_config")
def main(cfg: Config) -> None:
    test_data = load_dataset(cfg.data.path, split="test")
    test_data = test_data.filter(filter_by_origin, fn_kwargs={"origin": cfg.data.split}, num_proc=NUM_WORKERS, desc=f"Filtering {cfg.data.split} data")
    log.info(f"Loaded {len(test_data)} instances from {cfg.data.split}")
    target_size = int(len(test_data) * cfg.data.subset)
    if target_size % 2:
        target_size += 1
    test_data = test_data.sort(["instance_id", "verdict"]).select(range(target_size))
    log.info(f"Subsampled to {len(test_data)} instances")
    llm = LLM(
        model=cfg.model.name,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=1024,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)

    sampling_params_for_reasoning = SamplingParams(
        temperature=0.6,
        n=N,
        max_tokens=8192,
        stop="</reason>",
    )

    reasoning_responses = llm.chat(
        test_data["prompt"],
        sampling_params=sampling_params_for_reasoning,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    reasoning_responses = [[x.text for x in y.outputs] for y in reasoning_responses]

    # Construct responses with "Yes" answer
    yes_prompts = [
        existing_prompt[:-1]
        + [
            {
                "role": "assistant",
                "content": f"<reason>\n{x}\n</reason>\n<solution>\nYes\n</solution>",
            }
        ]
        for existing_prompt, y in zip(test_data["prompt"], reasoning_responses)
        for x in y
    ]
    # Construct responses with "No" answer
    no_prompts = [
        existing_prompt[:-1]
        + [
            {
                "role": "assistant",
                "content": f"<reason>\n{x}\n</reason>\n<solution>\nNo\n<solution>",
            }
        ]
        for existing_prompt, y in zip(test_data["prompt"], reasoning_responses)
        for x in y
    ]

    sampling_params_for_solution = SamplingParams(
        temperature=0,
        max_tokens=1,
        prompt_logprobs=1,
    )
    # Get responses for "Yes" and "No" prompts together for efficiency
    responses = llm.chat(yes_prompts + no_prompts, sampling_params_for_solution, continue_final_message=True, add_generation_prompt=False)

    # Split the responses into "Yes" and "No" responses
    yes_responses = responses[: len(yes_prompts)]
    no_responses = responses[len(yes_prompts) :]

    # Ensure that the responses are grouped by N
    yes_responses = [yes_responses[i : i + N] for i in range(0, len(yes_responses), N)]
    no_responses = [no_responses[i : i + N] for i in range(0, len(no_responses), N)]

    # Find the index of the "Yes" and "No" tokens in the responses
    yes_token_id = tokenizer.convert_tokens_to_ids("Yes")
    yes_indices = [[last_index(x.prompt_token_ids, yes_token_id) for x in y] for y in yes_responses]

    no_token_id = tokenizer.convert_tokens_to_ids("No")
    no_indices = [[last_index(x.prompt_token_ids, no_token_id) for x in y] for y in no_responses]

    # Obtain the log probabilities for the "Yes" and "No" tokens
    yes_logprobs = [[x.prompt_logprobs[yes_idx].get(yes_token_id).logprob for x, yes_idx in zip(y, yes_idx_list)] for y, yes_idx_list in zip(yes_responses, yes_indices)]

    no_logprobs = [[x.prompt_logprobs[no_idx].get(no_token_id).logprob for x, no_idx in zip(y, no_idx_list)] for y, no_idx_list in zip(no_responses, no_indices)]

    output_dir = Path(__file__).parent.parent / cfg.output.dir / cfg.data.split
    output_dir.mkdir(parents=True, exist_ok=True)
    # Initialize the results DataFrame
    if (output_dir / "scores_by_model.csv").exists():
        existing_results = pl.read_csv(output_dir / "scores_by_model.csv")
    else:
        existing_results = pl.DataFrame(schema=[("model", pl.String), ("K", pl.Int64), ("score", pl.Float64)])

    # Perform self-consistency at K for K = 1,2,4,8,...N
    for i in range(int(np.log2(N)) + 1):
        K = 2**i
        log.info(f"Evaluating with Self Consistency K={K}")
        # Compute the scores for each instance. The score is the P("Yes")/(P("Yes") + P("No"))
        scores_per_n = [[compute_score(yes_lp, no_lp) for yes_lp, no_lp in zip(yes_lp_n[:K], no_lp_n[:K])] for yes_lp_n, no_lp_n in zip(yes_logprobs, no_logprobs)]
        # Average the scores across the N responses for each prompt
        scores_per_prompt = [np.mean(scores) for scores in scores_per_n]

        test_data_K = test_data.add_column("score", scores_per_prompt)
        df = pl.from_arrow(test_data_K.data.table)
        df = df.with_columns(
            (pl.col("score").filter(pl.col("verdict") == "Yes").first().over("instance_id") > pl.col("score").filter(pl.col("verdict") == "No").first().over("instance_id")).cast(pl.Int64).alias("final_score"),
        )
        df = df.select(["instance_id", "score", "verdict", "final_score"])
        # Write prompt-wise results to CSV
        df.write_csv(output_dir / f"{cfg.model.name.split('/')[-1]}_K{K}.csv")

        # Create new row ensuring types match the schema
        new_row = pl.DataFrame(
            {
                "model": pl.Series([cfg.model.name], dtype=pl.String),
                "K": pl.Series([K], dtype=pl.Int64),
                "score": pl.Series([df["final_score"].mean()], dtype=pl.Float64),
            }
        )
        existing_results = pl.concat([existing_results, new_row])
        log.info(f"Average score for K={K}: {df['final_score'].mean()}")
    existing_results.write_csv(output_dir / "scores_by_model.csv")


if __name__ == "__main__":
    main()
