import logging
import os
from functools import partial, update_wrapper

import hydra
import rewards
import torch
import wandb
from configs.schema import Config
from datasets import load_dataset
from kernels import has_kernel
from omegaconf import OmegaConf
from prompts import LIST_REWARD_PROMPT, LIST_REWARD_PROMPT_COT
from transformers import AutoTokenizer
from utils import maybe_resume_training

from trl import GRPOConfig, GRPOTrainer

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
wandb.login()
os.environ["WANDB_ENTITY"] = "Aletheia"
os.environ["WANDB_PROJECT"] = "GRPO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def create_prompts(example, thinking=True):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    if thinking:
        example["prompt"] = [
            {
                "role": "user",
                "content": LIST_REWARD_PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                    valid_options=", ".join(potential_answers),
                ).strip(),
            },
        ]
    else:
        example["prompt"] = [
            {"role": "system", "content": LIST_REWARD_PROMPT_COT.format(valid_options=", ".join(potential_answers))},
            {
                "role": "user",
                "content": f"Here is the coding question followed by the candidate solutions:\n[QUESTION]\n{example['query']}\n[/QUESTION]\n\n{candidate_str}\n\nYour response should be exactly in the specified format, without any extra characters or spaces. Anything else will be considered invalid.",
            },
        ]

    if thinking:
        example["prompt"].append({"role": "assistant", "content": "<think>\n"})
    return example


def _filter_pairs(example):
    return example["num_candidates"] == 2


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: Config):
    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    model_short_name = cfg.grpo_params.model_path.split("/")[-1].lower()
    wandb_run_name = f"GRPO-{model_short_name}-{cfg.grpo_reward_type}-so{cfg.grpo_params.generate_every}"
    output_dir = f"{os.getenv('WORK')}/grpo_output/{wandb_run_name}"
    is_thinking_model = "deepseek" in cfg.grpo_params.model_path.lower()
    if is_thinking_model:
        REWARD_FUNC = [rewards.list_reward, rewards.list_format_reward]
    else:
        REWARD_FUNC = [rewards.list_reward_cot, rewards.list_format_reward_cot]

    if cfg.grpo_params.loss_type in ["dapo", "bnpo"]:
        soft_overlong = partial(
            rewards.soft_overlong_punishment, L_max=cfg.gen_params.max_completion_length, L_cache=1024 if cfg.gen_params.max_completion_length <= 4096 else 2048
        )
        update_wrapper(soft_overlong, rewards.soft_overlong_punishment)
        REWARD_FUNC.append(soft_overlong)

    if cfg.grpo_params.kl_penalty == "dynamic" and cfg.grpo_params.ref_model_sync_steps <= 0:
        raise ValueError("kl_update_steps must be greater than 0 for dynamic KL penalty.")

    if cfg.data.train.endswith(".parquet"):
        train_data = load_dataset("parquet", data_files=cfg.data.train)["train"]
    else:
        train_data = load_dataset(cfg.data.train)["train"]

    train_data = train_data.map(create_prompts, fn_kwargs={"thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")
    eval_data = None
    if cfg.data.val:
        if cfg.data.val.endswith(".parquet"):
            eval_data = load_dataset("parquet", data_files=cfg.data.val)["train"]
        else:
            eval_data = load_dataset(cfg.data.val)["train"]
        eval_data = eval_data.map(create_prompts, fn_kwargs={"thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")

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
        use_liger_loss=False,
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
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
