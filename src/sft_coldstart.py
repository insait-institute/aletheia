import logging
import os
from pathlib import Path

import hydra
import wandb
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

from configs.sft_coldstart_config import Config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
wandb.login()
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_PROJECT"] = "CodeShield"


def load_data(data_path: str) -> Dataset:
    data = load_dataset("parquet", data_files={"train": data_path})
    return data["train"]


def create_splits(data: Dataset, split_ratio: float = 0.8) -> tuple:
    data = data.shuffle(seed=42).select(range(100))
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
    config = SFTConfig(
        output_dir=cfg.sft_params.output_dir,
        overwrite_output_dir=cfg.sft_params.overwrite_output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=cfg.sft_params.batch_size,
        per_device_eval_batch_size=cfg.sft_params.batch_size,
        gradient_accumulation_steps=cfg.sft_params.gradient_accumulation_steps,
        learning_rate=cfg.sft_params.learning_rate,
        weight_decay=cfg.sft_params.weight_decay,
        warmup_steps=cfg.sft_params.warmup_steps,
        num_train_epochs=cfg.sft_params.num_epochs,
        logging_steps=cfg.sft_params.logging_steps,
        save_steps=cfg.sft_params.save_steps,
        report_to="wandb",
        run_name=cfg.wandb_params.run_name,
    )

    trainer = SFTTrainer(
        model_name=cfg.sft_config.model_name,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=config,
    )
    if isinstance(cfg.sft_params.output_dir, Path):
        cfg.sft_params.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.train()
    wandb.finish()
    log.info(f"Model trained and saved to {cfg.sft_params.output_dir}")


if __name__ == "__main__":
    train()
