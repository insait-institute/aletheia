import logging
import os
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from configs.sft_coldstart_config import Config

CHAT_TEMPLATE = """
{%- for message in messages %}
    {{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '\\n' + eos_token + '\\n'}}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n<reason>'}}
{%- endif %}
"""


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_logger():
    logger = logging.getLogger()
    if is_main_process():
        wandb.login()
        logging.basicConfig(level=logging.INFO)
        logger.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.CRITICAL)
        logger.setLevel(logging.CRITICAL)
    return logger


log = setup_logger()
dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=10))
os.environ["WANDB_PROJECT"] = "CodeShield"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OrderedSFTTrainer(SFTTrainer):
    def _get_train_sampler(self):
        return SequentialSampler(self.train_dataset)


class SampleInputCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if is_main_process():
            log.info("Sample input for training:")
            sample_input = self.trainer.get_train_dataloader().dataset[0]
            log.info(f"Sample input: {sample_input}")
            log.info(f"Sample input tokenized: {self.trainer.processing_class.apply_chat_template(sample_input['messages'], tokenize=False)}")


# def tokenize(examples, tokenizer, max_length):
#     formatted_texts = []
#     for message_chain in examples["messages"]:
#         text_parts = []
#         for message in message_chain:
#             text_parts.append(f"<|im_start|>{message['role']}\n{message['content']}\n<|im_end|>")
#         formatted_texts.append("\n".join(text_parts) + tokenizer.eos_token)
#     processed = tokenizer(text=formatted_texts, padding="max_length", max_length=max_length, truncation=True, add_special_tokens=True)
#     return processed


@hydra.main(version_base=None, config_name="sft_coldstart_config")
def train(cfg: Config) -> None:
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()
    model_short_name = cfg.sft_params.model_name.split("/")[-1].lower()
    output_dir = (Path(__file__).parent.parent / f"coldstart_output/{model_short_name}").as_posix()
    hub_model_id = f"CodeShield/sft-{model_short_name}-pad"

    train_data = load_dataset(cfg.data.path, split="train", num_proc=cpu_count)
    stage_1 = train_data.filter(lambda x: x["currciulum_stage"] == "stage_1", num_proc=os.cpu_count())
    stage_2 = train_data.filter(lambda x: x["currciulum_stage"] == "stage_2", num_proc=os.cpu_count())
    stage_1 = stage_1.shuffle(seed=cfg.sft_params.seed)
    stage_2 = stage_2.shuffle(seed=cfg.sft_params.seed)
    train_data = concatenate_datasets([stage_1, stage_2])

    log.info(f"Loaded data from {cfg.data.path}")
    if cfg.sft_params.max_length:
        log.info(f"Filtered data to max length {cfg.sft_params.max_length}")
    log.info(f"Train data size: {len(train_data)}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {cpu_count}")
    log.info(f"Number of GPUs: {gpu_count}")

    # Updated model loading to use explicit BF16 tensor type
    config = SFTConfig(
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
        log_level_replica="critical",
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
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        # Miscellaneous parameters
        ddp_backend="nccl",
        ddp_timeout=36000,
        deepspeed=cfg.sft_params.deepspeed_config_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft_params.model_name)
    tokenizer.chat_template = CHAT_TEMPLATE.strip()
    tokenizer.add_special_tokens(
        {
            "eos_token": "<|im_end|>",
            "pad_token": "<|end_of_text|>",
        }
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft_params.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    # train_data = train_data.map(tokenize, fn_kwargs={"tokenizer": tokenizer, "max_length": cfg.sft_params.max_length}, desc="Tokenizing data", batched=True, remove_columns=train_data.column_names)
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    # log.info(f"Sample tokenized data: {tokenizer.decode(train_data[0]['input_ids'])}")
    trainer = OrderedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_data,
        processing_class=tokenizer,
        data_collator=collator,
    )
    sample_input_callback = SampleInputCallback(trainer=trainer)
    trainer.add_callback(sample_input_callback)
    log.info(f"Using Training sampler: {trainer.get_train_dataloader().sampler.__class__.__name__}")
    trainer.train()
    log.info("Training completed.")
    trainer.save_model(output_dir)
    trainer.push_to_hub()
    log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
