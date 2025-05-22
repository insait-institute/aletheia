import logging
import os
import re
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import wandb
from configs.grpo_config import Config

dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=10))
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
wandb.login()
os.environ["WANDB_PROJECT"] = "CodeShield"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_data(data_path: str) -> Dataset:
    if data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files={"train": data_path})
    else:
        data = load_dataset(data_path)
    data = data["train"].shuffle(seed=42)
    return data


def create_splits(data: Dataset, split_ratio: float) -> tuple:
    if split_ratio == 1.0:
        return data, None
    train_size = int(len(data) * split_ratio)
    train_data = data.select(range(train_size))
    eval_data = data.select(range(train_size, len(data)))
    return train_data, eval_data


def check_valid_checkpoint(output_dir: str) -> bool | str:
    output_dir = Path(output_dir)
    try:
        checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    except Exception as e:
        log.error(f"Error checking for checkpoints: {e}")
        return None
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    return checkpoints[-1].as_posix()


def grpo_reward(completions, verdict, **kwargs):
    matches = [re.search(r"<solution>(.*?)</solution>", completion, re.DOTALL) for completion in completions]
    contents = [match.group(1).strip() if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, verdict)]


@hydra.main(version_base=None, config_name="grpo_config")
def train(cfg: Config):
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()
    model_short_name = cfg.grpo_params.model_name.split("/")[-1].lower()
    output_dir = (Path(__file__).parent.parent / f"grpo_output/{model_short_name}").as_posix()
    hub_model_id = f"CodeShield/grpo-{model_short_name}"

    data = load_data(cfg.data.path)
    train_data, eval_data = create_splits(data, cfg.data.split_ratio)

    log.info(f"Loaded data from {cfg.data.path}")
    if cfg.grpo_params.max_length:
        log.info(f"Filtered data to max length {cfg.grpo_params.max_length}")
    log.info(f"Data size: {len(data)}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {cpu_count}")
    log.info(f"Number of GPUs: {gpu_count}")

    # Updated model loading to use explicit BF16 tensor type
    checkpoint = check_valid_checkpoint(f"{output_dir}/intermediate_checkpoints")
    if checkpoint and cfg.grpo_params.resume_training_if_possible:
        log.info(f"Resuming training from checkpoint: {checkpoint}")

    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        resume_from_checkpoint=checkpoint if cfg.grpo_params.resume_training_if_possible else None,
        overwrite_output_dir=cfg.grpo_params.overwrite_output_dir,
        # GRPO parameters
        beta=cfg.grpo_params.beta,
        epsilon=cfg.grpo_params.epsilon,
        learning_rate=cfg.grpo_params.learning_rate,
        scale_rewards=not cfg.grpo_params.loss_type == "dr_grpo",
        loss_type=cfg.grpo_params.loss_type,
        # Training parameters
        num_train_epochs=cfg.grpo_params.num_epochs,
        eval_strategy="epoch" if eval_data else "no",
        save_strategy="epoch" if eval_data else "no",
        max_prompt_length=cfg.grpo_params.max_prompt_length,
        per_device_train_batch_size=cfg.grpo_params.batch_size,
        per_device_eval_batch_size=cfg.grpo_params.batch_size,
        gradient_accumulation_steps=cfg.grpo_params.gradient_accumulation_steps,
        bf16=cfg.grpo_params.use_bf16,
        lr_scheduler_type=cfg.grpo_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.grpo_params.lr_scheduler_kwargs,
        weight_decay=cfg.grpo_params.weight_decay,
        warmup_ratio=cfg.grpo_params.warmup_ratio,
        seed=cfg.grpo_params.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging parameters
        report_to="wandb",
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=False,
        run_name=cfg.wandb_params.run_name,
        logging_steps=cfg.grpo_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        hub_model_id=hub_model_id,
        hub_private_repo=True,
        # Data parameters
        data_seed=cfg.grpo_params.seed,
        dataloader_num_workers=cpu_count // gpu_count,
        remove_unused_columns=True,
        dataloader_drop_last=True,
        # Generation parameters
        use_vllm=True,
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
        vllm_gpu_memory_utilization=cfg.gen_params.vllm_gpu_memory_utilization,
        vllm_max_model_len=cfg.gen_params.vllm_max_model_len,
        vllm_dtype=cfg.gen_params.vllm_dtype,
        temperature=cfg.gen_params.temperature,
        max_completion_length=cfg.gen_params.max_completion_length,
        num_generations=cfg.gen_params.num_generations,
        # Miscellaneous parameters
        ddp_backend="nccl",
        ddp_timeout=36000,
        deepspeed=cfg.grpo_params.deepspeed_config_path,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.grpo_params.model_name)

    trainer = GRPOTrainer(
        model=cfg.grpo_params.model_name,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        reward_funcs=grpo_reward,
    )

    # Start training with explicit checkpoint resumption
    trainer.train(resume_from_checkpoint=checkpoint if cfg.grpo_params.resume_training_if_possible else None)
    log.info("Training completed.")
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
