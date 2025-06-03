import logging
import os
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from datasets import load_dataset
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


def grpo_reward(completions, verdict, **kwargs):
    contents = [completion.split("<solution>")[-1].split("</solution")[0].strip() for completion in completions]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, verdict)]


def convert_messages_to_chatml_strings(example):
    chatml = ""
    for dct in example["messages"]:
        chatml += f"<|im_start|>{dct['role']}\n{dct['content']}\n<|im_end|>\n"
    example["text"] = chatml.strip()
    return example


@hydra.main(version_base=None, config_name="grpo_config")
def train(cfg: Config):
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()
    model_short_name = cfg.grpo_params.model_name.split("/")[-1].lower()
    output_dir = (Path(__file__).parent.parent / f"grpo_output/{model_short_name}").as_posix()
    hub_model_id = f"CodeShield/grpo-{model_short_name}"

    data = load_dataset(cfg.data.path)
    train_data, eval_data = data["train"], None
    if "test" in data:
        eval_data = data["test"]
    log.info(f"Loaded data from {cfg.data.path}")
    if cfg.grpo_params.max_prompt_length:
        log.info(f"Filtered data to max length {cfg.grpo_params.max_prompt_length}")
    log.info(f"Data size: {len(data)}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {cpu_count}")
    log.info(f"Number of GPUs: {gpu_count}")

    # Updated model loading to use explicit BF16 tensor type
    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        # GRPO parameters
        beta=cfg.grpo_params.beta,
        epsilon=cfg.grpo_params.epsilon,
        learning_rate=cfg.grpo_params.learning_rate,
        loss_type=cfg.grpo_params.loss_type,
        scale_rewards=not cfg.grpo_params.loss_type == "dr_grpo",
        # Training parameters
        bf16=cfg.grpo_params.use_bf16,
        gradient_accumulation_steps=cfg.grpo_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=cfg.grpo_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.grpo_params.lr_scheduler_kwargs,
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
        hub_model_id=hub_model_id,
        hub_private_repo=True,
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=False,
        logging_steps=cfg.grpo_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="wandb",
        run_name=cfg.wandb_params.run_name,
        # Data parameters
        data_seed=cfg.grpo_params.seed,
        dataloader_num_workers=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else cpu_count,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        # Generation parameters
        max_completion_length=cfg.gen_params.max_completion_length,
        num_generations=cfg.gen_params.num_generations,
        temperature=cfg.gen_params.temperature,
        use_vllm=True,
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
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
        reward_funcs=grpo_reward,
    )

    # Start training with explicit checkpoint resumption
    trainer.train()
    log.info("Training completed.")
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
