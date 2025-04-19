import logging
import os

import hydra
import torch
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


def load_data(data_path: str) -> Dataset:
    if data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files={"train": data_path})
    else:
        data = load_dataset(data_path)
    return data["train"]


def create_splits(data: Dataset, split_ratio: float = 0.9) -> tuple:
    data = data.shuffle(seed=42)
    train_size = int(len(data) * split_ratio)
    train_data = data.select(range(train_size))
    eval_data = data.select(range(train_size, len(data)))
    return train_data, eval_data


@hydra.main(version_base=None, config_name="sft_coldstart_config")
def train(cfg: Config):
    data = load_data(cfg.data.path)
    train_data, eval_data = create_splits(data)
    log.info(f"Loaded data from {cfg.data.path}")
    log.info(f"Data size: {len(data)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}. Number of GPUs: {torch.cuda.device_count()}")
    model = AutoModelForCausalLM.from_pretrained(cfg.sft_params.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_params.model_name)

    config = SFTConfig(
        output_dir=f"{cfg.sft_params.output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.sft_params.overwrite_output_dir,
        num_train_epochs=cfg.sft_params.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.sft_params.batch_size,
        per_device_eval_batch_size=cfg.sft_params.batch_size,
        gradient_accumulation_steps=cfg.sft_params.gradient_accumulation_steps,
        bf16=cfg.sft_params.use_bf16,
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
        hub_model_id=cfg.sft_params.hub_model_id,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
    )
    trainer.train()
    log.info("Training completed.")
    trainer.save_model(cfg.sft_params.output_dir)
    trainer.push_to_hub(blocking=True)
    wandb.finish()
    log.info(f"Model trained and saved to {cfg.sft_params.output_dir}")


if __name__ == "__main__":
    train()
