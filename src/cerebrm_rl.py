import logging
import os
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import cerebrm_rewards
import wandb
from cerebrm_prompts import DS_GRM_PROMPT, JUDGELRM_PROMPT, LIST_REWARD_PROMPT
from configs.schema import Config

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


def _create_prompts(example, reward_type):
    potential_answers = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"][: example["num_candidates"]]
    candidates = [f"[RESPONSE_{i[2]}]\n{candidate}\n[/RESPONSE_{i[2]}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    if reward_type == "list_em" or reward_type == "list_dist":
        example["prompt"] = [
            {
                "role": "user",
                "content": LIST_REWARD_PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                    valid_options=", ".join(potential_answers),
                ),
            },
        ]
    elif reward_type == "judge_lrm":
        example["prompt"] = [
            {
                "role": "user",
                "content": JUDGELRM_PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                ),
            },
        ]
    elif reward_type == "ds_grm":
        example["prompt"] = [
            {
                "role": "user",
                "content": DS_GRM_PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                ),
            },
        ]
    return example


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: Config):
    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    model_short_name = cfg.grpo_params.model_path.split("/")[-1].lower()
    wandb_run_name = f"CerebRM-{model_short_name}-{cfg.reward_type}"
    output_dir = (Path(__file__).parent.parent / "cerebrm_output" / wandb_run_name).as_posix()

    if cfg.reward_type == "list_em":
        REWARD_FUNC = [cerebrm_rewards.list_reward, cerebrm_rewards.format_reward]
    elif cfg.reward_type == "list_dist":
        REWARD_FUNC = [cerebrm_rewards.list_reward_with_distance, cerebrm_rewards.format_reward]
    elif cfg.reward_type == "judge_lrm":
        REWARD_FUNC = [cerebrm_rewards.judgelrm_content_reward, cerebrm_rewards.judgelrm_format_reward]
    elif cfg.reward_type == "ds_grm":
        REWARD_FUNC = [cerebrm_rewards.list_score_correctness, cerebrm_rewards.list_score_max10, cerebrm_rewards.format_reward]
    else:
        raise ValueError(f"Unknown reward type: {cfg.reward_type}. Choose from 'list_em', 'list_dist', 'judge_lrm', 'ds_grm'.")

    if cfg.grpo_params.loss_type == "dapo":
        REWARD_FUNC.append(cerebrm_rewards.soft_overlong_punishment)

    if cfg.grpo_params.kl_penalty == "dynamic" and cfg.grpo_params.ref_model_sync_steps <= 0:
        raise ValueError("kl_update_steps must be greater than 0 for dynamic KL penalty.")

    train_data = load_dataset(cfg.data.train)["train"]
    if cfg.reward_type == "judge_lrm":
        train_data = train_data.filter(_filter_pairs, num_proc=NUM_WORKERS, desc="Only keeping pairs")

    eval_data = load_dataset(cfg.data.val)["Full"] if cfg.data.val else None
    train_data = train_data.map(_create_prompts, fn_kwargs={"reward_type": cfg.reward_type}, num_proc=NUM_WORKERS, desc="Creating prompts")
    log.info(f"Loaded data from {cfg.data.train}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {NUM_WORKERS}")
    log.info(f"Number of GPUs: {os.environ.get('WORLD_SIZE', torch.cuda.device_count())}")

    # Updated model loading to use explicit BF16 tensor type
    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16 if cfg.grpo_params.use_bf16 else torch.float32},
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
        use_liger_loss=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
        vllm_tensor_parallel_size=cfg.gen_params.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=cfg.gen_params.vllm_gpu_memory_utilization,
        # Miscellaneous parameters
        ddp_backend="nccl",
        ddp_timeout=36000,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.grpo_params.model_path)
    trainer = GRPOTrainer(
        model=cfg.grpo_params.model_path,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNC,
    )
    # Start training with explicit checkpoint resumption
    trainer.train()
    log.info("Training completed.")
    trainer.save_model(output_dir)
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
