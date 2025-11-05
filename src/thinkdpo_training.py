import logging
import os
from pathlib import Path

import hydra
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from trl import DPOConfig, DPOTrainer

import wandb
from configs.schema import Config
from utils import maybe_resume_training

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-DPO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def train_model(
    cfg: Config,
    model_name: str,
    data: Dataset,
    wandb_run_name: str,
    output_dir: str,
) -> None:
    config = DPOConfig(
        model_init_kwargs={"attn_implementation": "kernels-community/flash-attn"},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.dpo_params.overwrite_output_dir,
        # DPO Parameters
        beta=cfg.dpo_params.beta,
        sync_ref_model=cfg.dpo_params.sync_ref_model,
        ref_model_mixup_alpha=cfg.dpo_params.ref_model_mixup_alpha,
        ref_model_sync_steps=cfg.dpo_params.ref_model_sync_steps,
        # Training parameters
        bf16=cfg.dpo_params.use_bf16,
        eval_strategy="no",
        eval_steps=None,
        gradient_accumulation_steps=cfg.dpo_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.dpo_params.learning_rate,
        lr_scheduler_type=cfg.dpo_params.lr_scheduler_type,
        max_length=cfg.dpo_params.max_length,
        num_train_epochs=cfg.dpo_params.num_epochs,
        per_device_train_batch_size=cfg.dpo_params.batch_size,
        per_device_eval_batch_size=None,
        seed=cfg.dpo_params.seed,
        warmup_ratio=cfg.dpo_params.warmup_ratio,
        weight_decay=cfg.dpo_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.dpo_params.logging_steps,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"wetsoledrysoul/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.dpo_params.save_steps,
        # Data parameters
        data_seed=cfg.dpo_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_loss=True,
    )
    trainer = DPOTrainer(model=model_name, args=config, train_dataset=data)
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def does_file_exist(file: Path) -> bool:
    return file.exists()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    model_short_name = cfg.dpo_params.model_path.split("/")[-1]
    wandb_run_name = f"dpo_{model_short_name}"
    output_dir = Path(f"{os.getenv('WORK')}/dpo_output/{wandb_run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "model.safetensors.index.json", "generation_config.json"]]):
        log.info(f"dpo training files are already present in {output_dir}. Skipping.")
        return None

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    train_data = load_dataset(cfg.data.train)["train"]

    log.info(f"Training {cfg.dpo_params.model_path} on {len(train_data)} examples")
    train_model(cfg, cfg.dpo_params.model_path, train_data, wandb_run_name, output_dir)
    log.info(f"Completed training {cfg.dpo_params.model_path} on {len(train_data)} examples")


if __name__ == "__main__":
    main()
