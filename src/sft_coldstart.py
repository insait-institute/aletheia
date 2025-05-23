import logging
import os
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from datasets import load_dataset
from torch.utils.data import SequentialSampler
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from configs.sft_coldstart_config import Config

dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=10))
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
wandb.login()
os.environ["WANDB_PROJECT"] = "CodeShield"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OrderedSFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=SequentialSampler(self.train_dataset),
            is_training=True,
        )


@hydra.main(version_base=None, config_name="sft_coldstart_config")
def train(cfg: Config) -> None:
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()
    model_short_name = cfg.sft_params.model_name.split("/")[-1].lower()
    output_dir = (Path(__file__).parent.parent / f"coldstart_output/{model_short_name}").as_posix()
    hub_model_id = f"CodeShield/sft-{model_short_name}"

    train_data = load_dataset(cfg.data.path, split="train", num_proc=cpu_count)
    train_data = train_data.sort("currciulum_stage")
    log.info(f"Loaded data from {cfg.data.path}")
    if cfg.sft_params.max_length:
        log.info(f"Filtered data to max length {cfg.sft_params.max_length}")
    log.info(f"Train data size: {len(train_data)}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {cpu_count}")
    log.info(f"Number of GPUs: {gpu_count}")

    # Updated model loading to use explicit BF16 tensor type
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.sft_params.overwrite_output_dir,
        # Training parameters
        bf16=cfg.sft_params.use_bf16,
        eval_strategy="no",
        gradient_accumulation_steps=cfg.sft_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.sft_params.learning_rate,
        lr_scheduler_type=cfg.sft_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.sft_params.lr_scheduler_kwargs,
        max_length=cfg.sft_params.max_length,
        num_train_epochs=cfg.sft_params.num_epochs,
        per_device_train_batch_size=cfg.sft_params.batch_size,
        per_device_eval_batch_size=cfg.sft_params.batch_size,
        seed=cfg.sft_params.seed,
        warmup_ratio=cfg.sft_params.warmup_ratio,
        weight_decay=cfg.sft_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=False,
        logging_steps=cfg.sft_params.logging_steps,
        report_to="wandb",
        run_name=cfg.wandb_params.run_name,
        # Saving parameters
        hub_model_id=hub_model_id,
        hub_private_repo=True,
        save_strategy="steps",
        save_steps=cfg.sft_params.save_steps,
        # Data parameters
        data_seed=cfg.sft_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=cpu_count // gpu_count,
        remove_unused_columns=True,
        # Miscellaneous parameters
        ddp_backend="nccl",
        ddp_timeout=36000,
        deepspeed=cfg.sft_params.deepspeed_config_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_params.model_name)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=cfg.sft_params.model_name,
        args=config,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    log.info("Training completed.")
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
