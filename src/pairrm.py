import logging
import os
import re

# from datetime import timedelta
from pathlib import Path

import hydra
import torch

# import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from configs.pairrm_config import Config

# dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=10))
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# wandb.login()
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-GRPO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
DAPO_ARGS = {"L_max": 3072, "L_cache": 1024}


def correctness_reward(completions, correct_ans, **kwargs):
    contents = [completion[0]["content"].split("</think>")[-1].strip() for completion in completions]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, correct_ans)]


def format_reward(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>(.*?)$"
    completion_contents = [completion[0]["content"].split("<|im_end|>")[0].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.0 if match and match.group(1).strip() in ["[[A]]", "[[B]]"] else -1.0 for match in matches]


def soft_overlong_punishment(completion_ids, **kwargs):
    # taken from https://github.com/huggingface/trl/issues/3130
    rewards = []
    for ids in completion_ids:
        completion_length = len(ids)
        if completion_length <= DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"]:
            rewards.append(0)
        elif DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"] < completion_length <= DAPO_ARGS["L_max"]:
            rewards.append((DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"] - completion_length) / DAPO_ARGS["L_cache"])
        else:
            rewards.append(-1)
    return rewards


@hydra.main(version_base=None, config_name="grpo_config")
def train(cfg: Config):
    model_short_name = cfg.grpo_params.model_name.split("/")[-1].lower().replace("sft-", "")
    wandb_run_name = model_short_name + "testing"
    output_dir = (Path(__file__).parent.parent / "grpo_output" / wandb_run_name).as_posix()

    if cfg.grpo_params.reward_type == "correctness_only":
        REWARD_FUNC = [correctness_reward]
    elif cfg.grpo_params.reward_type == "correctness_format":
        REWARD_FUNC = [correctness_reward, format_reward]
    else:
        raise ValueError(f"Unknown reward type: {cfg.grpo_params.reward_type}. Choose from 'correctness_only', 'correctness_format'.")

    if cfg.grpo_params.loss_type == "bnpo":
        REWARD_FUNC.append(soft_overlong_punishment)

    if cfg.grpo_params.kl_penalty == "dynamic" and cfg.grpo_params.ref_model_sync_steps <= 0:
        raise ValueError("kl_update_steps must be greater than 0 for dynamic KL penalty.")

    data = load_dataset(cfg.data.path)
    train_data, eval_data = data["train"], None
    # if "test" in data:
    #     eval_data = data["test"]
    log.info(f"Loaded data from {cfg.data.path}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {NUM_WORKERS}")
    log.info(f"Number of GPUs: {os.environ.get('WORLD_SIZE', torch.cuda.device_count())}")

    # Updated model loading to use explicit BF16 tensor type
    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        # GRPO parameters
        beta=0.0 if cfg.grpo_params.kl_penalty == "no" else cfg.grpo_params.beta,
        epsilon=cfg.grpo_params.epsilon,
        epsilon_high=cfg.grpo_params.epsilon_high if cfg.grpo_params.loss_type == "bnpo" else None,
        learning_rate=cfg.grpo_params.learning_rate,
        loss_type=cfg.grpo_params.loss_type,
        mask_truncated_completions=True,
        sync_ref_model=cfg.grpo_params.kl_penalty == "dynamic",
        ref_model_mixup_alpha=cfg.grpo_params.ref_model_syncup_alpha,
        ref_model_sync_steps=cfg.grpo_params.ref_model_sync_steps,
        scale_rewards=not cfg.grpo_params.loss_type == "dr_grpo",
        # Training parameters
        bf16=cfg.grpo_params.use_bf16,
        bf16_full_eval=cfg.grpo_params.use_bf16,
        gradient_accumulation_steps=cfg.grpo_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=cfg.grpo_params.lr_scheduler_type,
        max_prompt_length=cfg.grpo_params.max_prompt_length,
        num_train_epochs=cfg.grpo_params.num_epochs,
        per_device_train_batch_size=cfg.grpo_params.batch_size,
        seed=cfg.grpo_params.seed,
        weight_decay=cfg.grpo_params.weight_decay,
        warmup_ratio=cfg.grpo_params.warmup_ratio,
        # Evaluation parameters
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.grpo_params.save_steps if eval_data else None,
        per_device_eval_batch_size=cfg.grpo_params.eval_batch_size if eval_data else None,
        # Checkpointing parameters
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.grpo_params.overwrite_output_dir,
        save_strategy="no",
        save_steps=cfg.grpo_params.save_steps,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.grpo_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="none",
        # run_name=wandb_run_name,
        # Data parameters
        data_seed=cfg.grpo_params.seed,
        dataloader_num_workers=NUM_WORKERS,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        # Generation parameters
        max_completion_length=cfg.gen_params.max_completion_length,
        num_generations=cfg.gen_params.num_generations,
        temperature=cfg.gen_params.temperature,
        use_liger_loss=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
        vllm_tensor_parallel_size=cfg.gen_params.vllm_tensor_parallel_size,
        # Miscellaneous parameters
        ddp_backend="nccl",
        ddp_timeout=36000,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.grpo_params.model_name)
    trainer = GRPOTrainer(
        model=cfg.grpo_params.model_name,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNC,
    )
    dl = trainer.get_train_dataloader()
    log.info(f"DataLoader created with {len(dl)} batches, {type(dl)} and {vars(dl)} and {vars(dl.base_dataloader)}")
    log.info(f"Training dataset size: {len(trainer.train_dataset)}")
    # Start training with explicit checkpoint resumption
    trainer.train()
    log.info("Training completed.")
    trainer.save_model(output_dir)
    # trainer.push_to_hub()
    # log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
