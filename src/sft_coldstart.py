import logging
import os
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset, load_dataset
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
device_string = PartialState().process_index


def _dicts_to_chatml(example):
    chatml = ""
    for dct in example["messages"]:
        chatml += f"<|im_start|>{dct['role']}\n{dct['content']}\n<|im_end|>\n"
    return chatml.strip()


def load_data(data_path: str, max_length: int | None, subset: int | float) -> Dataset:
    if data_path.endswith(".parquet"):
        data = load_dataset("parquet", data_files={"train": data_path})
    else:
        data = load_dataset(data_path, token=os.getenv("HF_TOKEN"))
    data = data["train"].shuffle(seed=42)
    if max_length:
        data = data.filter(lambda x: x["num_tokens"] <= max_length)
    if subset:
        if isinstance(subset, float):
            subset = int(len(data) * subset)
        data = data.select(range(subset))
    return data


def create_splits(data: Dataset, split_ratio: float) -> tuple:
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


@hydra.main(version_base=None, config_name="sft_coldstart_config")
def train(cfg: Config):
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()
    model_short_name = cfg.sft_params.model_name.split("/")[-1].lower()
    output_dir = (Path(__file__).parent.parent / f"coldstart_output/{model_short_name}").as_posix()
    hub_model_id = f"CodeShield/sft-{model_short_name}"
    if not cfg.data.subset == 1.0:
        output_dir += f"_subset{cfg.data.subset}"
        hub_model_id += f"-subset{cfg.data.subset}"

    data = load_data(cfg.data.path, cfg.sft_params.max_length, cfg.data.subset)
    train_data, eval_data = create_splits(data, cfg.data.split_ratio)

    log.info(f"Loaded data from {cfg.data.path}")
    if cfg.sft_params.max_length:
        log.info(f"Filtered data to max length {cfg.sft_params.max_length}")
    log.info(f"Data size: {len(data)}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data)}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {cpu_count}")
    log.info(f"Number of GPUs: {gpu_count}")

    # Updated model loading to use explicit BF16 tensor type
    checkpoint = check_valid_checkpoint(f"{output_dir}/intermediate_checkpoints")
    if checkpoint and cfg.sft_params.resume_training_if_possible:
        log.info(f"Resuming training from checkpoint: {checkpoint}")

    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
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
        tp_size=cfg.sft_params.tp_size,
        bf16_full_eval=cfg.sft_params.use_bf16,
        learning_rate=cfg.sft_params.learning_rate,
        lr_scheduler_type=cfg.sft_params.lr_scheduler_type,
        weight_decay=cfg.sft_params.weight_decay,
        warmup_steps=cfg.sft_params.warmup_steps,
        report_to="wandb",
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=False,
        run_name=cfg.wandb_params.run_name,
        logging_steps=cfg.sft_params.logging_steps,
        seed=cfg.sft_params.seed,
        data_seed=cfg.sft_params.seed,
        load_best_model_at_end=True,
        hub_model_id=hub_model_id,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cpu_count // gpu_count,
        remove_unused_columns=True,
        dataloader_drop_last=True,
        ddp_backend="nccl",
        ddp_timeout=36000,
        deepspeed=cfg.sft_params.deepspeed_config_path,
    )

    # model = AutoModelForCausalLM.from_pretrained(cfg.sft_params.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_params.model_name)

    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=cfg.sft_params.model_name,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        formatting_func=_dicts_to_chatml,
        data_collator=collator,
    )

    # Start training with explicit checkpoint resumption
    trainer.train(resume_from_checkpoint=checkpoint if cfg.sft_params.resume_training_if_possible else None)
    log.info("Training completed.")
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    wandb.finish()
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
