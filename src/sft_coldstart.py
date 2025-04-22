import logging
import os
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from configs.sft_coldstart_config import Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
wandb.login()
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_PROJECT"] = "CodeShield"
device_string = PartialState().process_index
device_map = {"": device_string}


def load_data(data_path: str) -> Dataset:
    if data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files={"train": data_path})
    else:
        data = load_dataset(data_path, data_files={"train": "data/train-*"})
    return data["train"]


def create_splits(data: Dataset, split_ratio: float) -> tuple:
    data = data.shuffle(seed=42).select(range(1000))
    train_size = int(len(data) * split_ratio)
    train_data = data.select(range(train_size))
    eval_data = data.select(range(train_size, len(data)))
    return train_data, eval_data


def check_valid_checkpoint(output_dir: str) -> bool | str:
    output_dir = Path(output_dir)
    checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    return checkpoints[-1].as_posix()


@hydra.main(version_base=None, config_name="sft_coldstart_config")
def train(cfg: Config):
    model_short_name = cfg.sft_params.model_name.split("/")[-1].lower()
    output_dir = (Path(__file__).parent.parent / f"coldstart_output/{model_short_name}").as_posix()
    hub_model_id = f"CodeShield/{model_short_name}-sft"
    data = load_data(cfg.data.path)
    train_data, eval_data = create_splits(data, cfg.data.split_ratio)

    log.info(f"Loaded data from {cfg.data.path}")
    log.info(f"Data size: {len(data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}. Number of GPUs: {torch.cuda.device_count()}")
    model = AutoModelForCausalLM.from_pretrained(cfg.sft_params.model_name, device_map="auto", torch_dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_params.model_name)

    checkpoint = check_valid_checkpoint(f"{output_dir}/intermediate_checkpoints")
    if checkpoint and cfg.sft_params.resume_training_if_possible:
        log.info(f"Resuming training from checkpoint: {checkpoint}")
    config = SFTConfig(
        output_dir=f"{output_dir}/intermediate_checkpoints",
        resume_from_checkpoint=checkpoint if cfg.sft_params.resume_training_if_possible else None,
        overwrite_output_dir=cfg.sft_params.overwrite_output_dir,
        num_train_epochs=cfg.sft_params.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_length=cfg.sft_params.max_length,
        per_device_train_batch_size=cfg.sft_params.batch_size,
        per_device_eval_batch_size=cfg.sft_params.batch_size,
        gradient_accumulation_steps=cfg.sft_params.gradient_accumulation_steps,
        bf16=cfg.sft_params.use_bf16,
        bf16_full_eval=cfg.sft_params.use_bf16,
        learning_rate=cfg.sft_params.learning_rate,
        lr_scheduler_type=cfg.sft_params.lr_scheduler_type,
        weight_decay=cfg.sft_params.weight_decay,
        warmup_steps=cfg.sft_params.warmup_steps,
        report_to="wandb",
        log_level=cfg.wandb_params.log_level,
        run_name=cfg.wandb_params.run_name,
        logging_steps=cfg.sft_params.logging_steps,
        seed=cfg.sft_params.seed,
        data_seed=cfg.sft_params.seed,
        save_total_limit=1,
        load_best_model_at_end=True,
        hub_model_id=hub_model_id,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        ddp_broadcast_buffers=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=checkpoint if cfg.sft_params.resume_training_if_possible else None)
    log.info("Training completed.")
    trainer.save_model(output_dir)
    trainer.push_to_hub(blocking=True)
    wandb.finish()
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
