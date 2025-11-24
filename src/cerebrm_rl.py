import logging
import os

import hydra
import torch
from datasets import load_dataset
from kernels import has_kernel
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import cerebrm_rewards
import wandb
from configs.schema import Config
from utils import create_prompts, maybe_resume_training

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
wandb.login()
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-GRPO-0925"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _filter_pairs(example):
    return example["num_candidates"] == 2


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: Config):
    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    model_short_name = cfg.grpo_params.model_path.split("/")[-1].lower()
    wandb_run_name = f"CerebRM-{model_short_name}-{cfg.grpo_reward_type}"
    output_dir = f"{os.getenv('WORK')}/cerebrm_output/{wandb_run_name}"
    is_thinking_model = "deepseek" in cfg.grpo_params.model_path.lower()
    if cfg.grpo_reward_type in ["list_em", "pair"]:
        if is_thinking_model:
            REWARD_FUNC = [cerebrm_rewards.list_reward, cerebrm_rewards.list_format_reward]
        else:
            REWARD_FUNC = [cerebrm_rewards.list_reward_cot, cerebrm_rewards.list_format_reward_cot]
    elif cfg.grpo_reward_type == "list_dist":
        REWARD_FUNC = [cerebrm_rewards.list_reward_with_distance, cerebrm_rewards.list_format_reward]
    elif cfg.grpo_reward_type == "judge_lrm":
        REWARD_FUNC = [cerebrm_rewards.judgelrm_content_reward, cerebrm_rewards.judgelrm_format_reward]
    elif cfg.grpo_reward_type == "ds_grm":
        REWARD_FUNC = [cerebrm_rewards.grm_correctness_reward, cerebrm_rewards.grm_format_reward]
    else:
        raise ValueError(f"Unknown reward type: {cfg.grpo_reward_type}. Choose from 'list_em', 'list_dist', 'judge_lrm', 'ds_grm'.")

    if cfg.grpo_params.loss_type in ["dapo", "bnpo"]:
        REWARD_FUNC.append(cerebrm_rewards.soft_overlong_punishment)

    if cfg.grpo_params.kl_penalty == "dynamic" and cfg.grpo_params.ref_model_sync_steps <= 0:
        raise ValueError("kl_update_steps must be greater than 0 for dynamic KL penalty.")

    train_data = load_dataset(cfg.data.train)["train"]
    if cfg.grpo_reward_type in ["judge_lrm", "pair"]:
        train_data = train_data.filter(_filter_pairs, num_proc=NUM_WORKERS, desc="Only keeping pairs for judge_lrm")
    train_data = train_data.map(create_prompts, fn_kwargs={"grpo_reward_type": cfg.grpo_reward_type, "thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")
    if cfg.data.val:
        if cfg.data.val == cfg.data.train:
            eval_data = load_dataset(cfg.data.val)["test_weak_easy"]
        else:
            eval_data = load_dataset(cfg.data.val)["Full"]
        eval_data = eval_data.map(create_prompts, fn_kwargs={"grpo_reward_type": cfg.grpo_reward_type, "thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")
    else:
        eval_data = None
    log.info(f"Example prompt: {train_data['prompt'][0]}")
    log.info(f"Loaded data from {cfg.data.train}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {NUM_WORKERS}")
    log.info(f"Number of GPUs: {os.environ.get('WORLD_SIZE', torch.cuda.device_count())}")
    kernel = None
    if has_kernel("kernels-community/flash-attn3"):
        kernel = "kernels-community/flash-attn3"
        log.info("Flash Attention 3 kernel found. Using Flash Attention 3 for training.")
    elif has_kernel("kernels-community/flash-attn2"):
        kernel = "kernels-community/flash-attn2"
        log.info("Flash Attention 2 kernel found. Using Flash Attention 2 for training.")
    elif has_kernel("kernels-community/flash-attn"):
        kernel = "kernels-community/flash-attn"
        log.info("Flash Attention kernel found. Using Flash Attention for training.")

    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": kernel},
        # GRPO parameters
        beta=0.0 if cfg.grpo_params.kl_penalty == "no" else cfg.grpo_params.beta,
        epsilon=cfg.grpo_params.epsilon,
        epsilon_high=cfg.grpo_params.epsilon_high,
        learning_rate=cfg.grpo_params.learning_rate,
        loss_type=cfg.grpo_params.loss_type,
        mask_truncated_completions=True,
        sync_ref_model=cfg.grpo_params.kl_penalty == "dynamic",
        ref_model_mixup_alpha=cfg.grpo_params.ref_model_mixup_alpha,
        ref_model_sync_steps=cfg.grpo_params.ref_model_sync_steps,
        # Training parameters
        bf16=cfg.grpo_params.use_bf16,
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
        save_strategy="steps",
        save_steps=cfg.grpo_params.save_steps,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.grpo_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Data parameters
        data_seed=cfg.grpo_params.seed,
        dataloader_num_workers=NUM_WORKERS,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        # Generation parameters
        max_completion_length=cfg.gen_params.max_completion_length,
        num_generations=cfg.gen_params.num_generations,
        temperature=cfg.gen_params.temperature,
        use_liger_loss=(not cfg.grpo_params.importance_sampling_level == "sequence" and not cfg.grpo_params.loss_type == "dapo"),
        use_vllm=True,
        vllm_mode="colocate",
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
        vllm_tensor_parallel_size=cfg.gen_params.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=cfg.gen_params.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        # Changes for speedup
        steps_per_generation=cfg.grpo_params.gradient_accumulation_steps * cfg.grpo_params.generate_every,
        importance_sampling_level=cfg.grpo_params.importance_sampling_level,
        scale_rewards=cfg.grpo_params.scale_rewards,
    )

    if cfg.grpo_use_lora:
        peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")

    tokenizer = AutoTokenizer.from_pretrained(cfg.grpo_params.model_path)
    trainer = GRPOTrainer(
        model=cfg.grpo_params.model_path,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNC,
        peft_config=peft_config if cfg.grpo_use_lora else None,
    )
    # Start training with explicit checkpoint resumption
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


if __name__ == "__main__":
    train()
