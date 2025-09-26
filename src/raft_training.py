import logging
import os
import random
from typing import Dict, List

import hydra
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from cerebrm_prompts import RAFT_PROMPT
from cerebrm_rewards import extract_boxed_contents_list
from configs.schema import Config
from utils import get_generated_text, run_inference

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-RAFT"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def is_valid_checkpoint(checkpoint_dir: str) -> bool:
    if not os.path.isdir(checkpoint_dir):
        return False

    required_files = ["config.json", "tokenizer.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]

    # Check required config + tokenizer
    for file in required_files:
        if not os.path.isfile(os.path.join(checkpoint_dir, file)):
            return False

    # At least one model weight file must exist
    if not any(os.path.isfile(os.path.join(checkpoint_dir, mf)) for mf in model_files):
        return False

    return True


def _create_prompts(example, cfg: Config):
    potential_answers = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i[2]}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i[2]}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    example["prompt"] = [
        {
            "role": "user",
            "content": RAFT_PROMPT.format(
                question=example["query"],
                candidates=candidate_str,
                valid_options=", ".join(potential_answers),
            ).strip(),
        },
    ]
    if cfg.raft_params.thinking_model:
        example["prompt"].append(
            {"role": "assistant", "content": "<think>\n"},
        )  # to avoid bug in reward calculation in older trl versions
    return example


def stage_one(cfg: Config, prompts: List[List[Dict[str, str]]], model_name_episode: str) -> List[List[str]]:
    # Sampling K responses from the current model

    responses = run_inference(prompts, model_name_episode, temperature=1.0, max_tokens=8192, n=cfg.raft_params.num_generations)
    completions = get_generated_text(responses)
    return completions


def stage_two(completions: List[List[str]], ground_truths: List[str]) -> List[List[bool]]:
    # Scoring the K responses using a verifiable reward
    model_answers = [[extract_boxed_contents_list(y) for y in x] for x in completions]
    scored_completions = [[model_ans == gt for model_ans in model_ans_list] for model_ans_list, gt in zip(model_answers, ground_truths)]
    return scored_completions


def stage_three(
    cfg: Config,
    model_name_episode: str,
    stage3_data: Dataset,
    wandb_run_name: str,
    output_dir: str,
    eval_data: Dataset | None,
) -> None:
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "kernels-community/flash-attn"},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.raft_params.overwrite_output_dir,
        assistant_only_loss=True,
        # Training parameters
        bf16=cfg.raft_params.use_bf16,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.raft_params.save_steps if eval_data else None,
        gradient_accumulation_steps=cfg.raft_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.raft_params.learning_rate,
        lr_scheduler_type=cfg.raft_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.raft_params.lr_scheduler_kwargs,
        max_length=cfg.raft_params.max_length,
        num_train_epochs=cfg.raft_params.num_epochs,
        per_device_train_batch_size=cfg.raft_params.batch_size,
        per_device_eval_batch_size=cfg.raft_params.batch_size,
        seed=cfg.raft_params.seed,
        warmup_ratio=cfg.raft_params.warmup_ratio,
        weight_decay=cfg.raft_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.raft_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"CodeShield/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.raft_params.save_steps,
        save_total_limit=cfg.raft_params.save_total_limit,
        # Data parameters
        data_seed=cfg.raft_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    trainer = SFTTrainer(model=model_name_episode, args=config, train_dataset=stage3_data)
    trainer.train(resume_from_checkpoint=is_valid_checkpoint(config.output_dir))
    trainer.push_to_hub()


def _count_tokens(prompt, tokenizer) -> int:
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized_prompt = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
    return len(tokenized_prompt)


def construct_data_for_stage3(stage1_prompts, episode_completions, scored_completions, tokenizer) -> Dataset:
    messages, num_tokens = [], []
    for prompt, completions, scores in zip(stage1_prompts, episode_completions, scored_completions):
        completions = [c for c, s in zip(completions, scores) if s]
        if not completions:
            continue
        completion = random.sample(completions, 1)[0]
        if prompt[-1]["role"] == "assistant":
            completion = prompt[-1]["content"] + completion
            prompt = prompt[:-1]
        messages.append(prompt + [{"role": "assistant", "content": completion}])
    for prompt in messages:
        num_tokens.append(_count_tokens(prompt, tokenizer))
    return Dataset.from_dict({"messages": messages, "num_tokens": num_tokens})


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: Config):
    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.raft_params.model_name)
    train_data = load_dataset(cfg.data.train)["train"]
    if cfg.data.val:
        eval_data = load_dataset(cfg.data.val)
        if "test_weak_easy" in eval_data:
            eval_data = eval_data["test_weak_easy"]
        elif "Full" in eval_data:
            eval_data = eval_data["Full"]
        else:
            eval_data = eval_data["test"]
    else:
        eval_data = None

    output_dir = f"{os.getenv('WORK')}/raft_output"
    model_short_name = cfg.raft_params.model_name.split("/")[-1]
    wandb_run_name = f"RAFT_{model_short_name}_ep{cfg.raft_params.episode_num}"
    model_name_episode = cfg.raft_params.model_name if cfg.raft_params.episode_num == 0 else f"{output_dir}/{wandb_run_name}"

    log.info(f"Starting RAFT episode {cfg.raft_params.episode_num}")

    stage1_prompts = train_data.map(_create_prompts, num_proc=NUM_WORKERS, desc="Creating prompts")
    log.info(f"Processing {len(stage1_prompts)} prompts for stage one")
    episode_completions = stage_one(stage1_prompts, model_name_episode)

    log.info(f"Scoring {len(episode_completions)} prompts for stage two")
    scored_completions = stage_two(episode_completions, list(train_data["chosen_answer"]))

    stage3_data = construct_data_for_stage3(train_data, episode_completions, scored_completions, tokenizer)
    log.info(f"Training {model_name_episode} on {len(stage3_data)} examples in stage three")
    stage_three(cfg, model_name_episode, stage3_data, wandb_run_name, output_dir, eval_data)

    log.info(f"Completed RAFT episode {cfg.raft_params.episode_num}")


if __name__ == "__main__":
    main()
