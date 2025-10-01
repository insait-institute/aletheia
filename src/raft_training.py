import logging
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from cerebrm_prompts import RAFT_PROMPT_NOTHINK, RAFT_PROMPT_THINK
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


@dataclass
class Message:
    role: str
    content: str


Prompt = List[Message]


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
    RAFT_PROMPT = RAFT_PROMPT_THINK if cfg.raft_params.thinking_model else RAFT_PROMPT_NOTHINK
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


def generate_completions(cfg: Config, prompts: List[List[Dict[str, str]]], model: str, tokenizer) -> List[List[str]]:
    # Sampling K responses from the current model
    responses = run_inference(prompts, model, temperature=1.0, max_tokens=8192, n=cfg.raft_params.num_generations, tp_size=torch.cuda.device_count(), tokenizer=tokenizer, max_model_len=12288)
    completions = get_generated_text(responses)
    return completions


def score_completions(completions: List[List[str]], ground_truths: List[str]) -> List[List[bool]]:
    # Scoring the K responses using a verifiable reward
    model_answers = [[extract_boxed_contents_list(y) for y in x] for x in completions]
    scored_completions = [[model_ans == gt for model_ans in model_ans_list] for model_ans_list, gt in zip(model_answers, ground_truths)]
    return scored_completions


def train_raft_model(
    cfg: Config,
    model_path_episode: str,
    stage3_data: Dataset,
    wandb_run_name: str,
    output_dir: str,
    eval_data: Dataset | None,
) -> None:
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "kernels-community/flash-attn"},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.raft_params.overwrite_output_dir,
        completion_only_loss=True,
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
    trainer = SFTTrainer(model=model_path_episode, args=config, train_dataset=stage3_data)
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


def construct_data_for_training(stage1_prompts, episode_completions, scored_completions, tokenizer) -> Dataset:
    prompts, completions, num_tokens = [], [], []
    for prompt, completions_list, scores in zip(stage1_prompts, episode_completions, scored_completions):
        completions_list = [c for c, s in zip(completions_list, scores) if s]
        if not completions_list:
            continue
        completion = random.sample(completions_list, 1)[0]
        if prompt[-1]["role"] == "assistant":
            completion = prompt[-1]["content"] + completion
            prompt = prompt[:-1]
        completion = [{"role": "assistant", "content": completion}]
        prompts.append(prompt)
        completions.append(completion)
    # print(type(prompts), type(completions), type(prompts[0]), type(completions[0]), prompts[0], completions[0])
    num_tokens = [_count_tokens(prompt + completion, tokenizer) for prompt, completion in zip(prompts, completions)]
    return Dataset.from_dict({"prompt": prompts, "completion": completions, "num_tokens": num_tokens})


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    ### Sanity checks
    assert cfg.raft_stage in [1, 2], "Invalid raft_stage. Must be 1 for generating and scoring completions, or 2 for training."
    assert cfg.raft_episode >= 0, "Invalid raft_episode. Must be a non-negative integer."
    output_dir = Path(f"{os.getenv('WORK')}/raft_output/episode_{cfg.raft_episode}")
    if cfg.raft_episode > 0:
        for prev_ep in range(cfg.raft_episode):
            prev_dir = output_dir.parent / f"episode_{prev_ep}"
            if "completions.pkl" not in prev_dir.iterdir() or "scored_completions.pkl" not in prev_dir.iterdir():
                raise ValueError(f"Previous episode directory {prev_dir} is incomplete. Please run stage 1 and 2 for episode {prev_ep} before proceeding.")
    ### End sanity checks
    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.raft_params.model_path)
    train_data = load_dataset(cfg.data.train)["train"]
    train_data = train_data.shuffle(seed=cfg.raft_params.seed).select(range(5120))
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

    output_dir.mkdir(parents=True, exist_ok=True)
    model_short_name = cfg.raft_params.model_path.split("/")[-1]
    wandb_run_name = f"RAFT_{model_short_name}_ep{cfg.raft_episode}"
    model_path_episode = cfg.raft_params.model_path if cfg.raft_episode == 0 else f"CodeShield/{wandb_run_name}"

    train_data = train_data.map(_create_prompts, fn_kwargs={"cfg": cfg}, num_proc=NUM_WORKERS, desc="Creating prompts")
    stage1_prompts = list(train_data["prompt"])
    log.info(f"Starting RAFT episode {cfg.raft_episode}")
    if cfg.raft_stage == 1:
        log.info(f"Processing {len(stage1_prompts)} prompts for stage one")
        episode_completions = generate_completions(cfg, stage1_prompts, model_path_episode, tokenizer)
        log.info(f"Scoring {len(episode_completions)} prompts for stage two")
        scored_completions = score_completions(episode_completions, list(train_data["chosen_answer"]))
        with open(output_dir / "completions.pkl", "wb") as f:
            pickle.dump(episode_completions, f)
        with open(output_dir / "scored_completions.pkl", "wb") as f:
            pickle.dump(scored_completions, f)
        log.info(f"Completed Stage 1 of RAFT episode {cfg.raft_episode}")

    else:
        with open(output_dir / "completions.pkl", "rb") as f:
            episode_completions = pickle.load(f)
        with open(output_dir / "scored_completions.pkl", "rb") as f:
            scored_completions = pickle.load(f)
        stage3_data = construct_data_for_training(stage1_prompts, episode_completions, scored_completions, tokenizer)
        log.info(f"Training {model_path_episode} on {len(stage3_data)} examples in Stage 2")
        train_raft_model(cfg, model_path_episode, stage3_data, wandb_run_name, output_dir, eval_data)
        log.info(f"Completed Stage 2 of RAFT episode {cfg.raft_episode}")


if __name__ == "__main__":
    """
    Usage guide:
    Since only one vllm object can be created at a time (without running into vllm errors), the script is designed to be run in two `sub_stages` per RAFT episode.
    For each episode, run the following commands sequentially:
    
    raft_training.py +raft_stage=1 raft_episode=0
    raft_training.py +raft_stage=2 raft_episode=0
    raft_training.py +raft_stage=1 raft_episode=1
    raft_training.py +raft_stage=2 raft_episode=1
    
    and so on.
    """
    main()
