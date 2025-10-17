import logging
import os
from pathlib import Path

import hydra
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from cerebrm_prompts import GENRM_PROMPT
from configs.schema import Config
from utils import Prompt, maybe_resume_training

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-GenRM"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_training_dataset(example):
    potential_answers = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i[2]}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i[2]}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    example["prompt"] = [
        {
            "role": "user",
            "content": GENRM_PROMPT.format(
                question=example["query"],
                candidates=candidate_str,
                valid_options=", ".join(potential_answers),
            ).strip(),
        },
    ]
    example["completion"] = [{"role": "assistant", "content": f"\\boxed{{{example['chosen_answer']}}}"}]
    return example


def train_model(
    cfg: Config,
    model_name: str,
    data: Dataset,
    wandb_run_name: str,
    output_dir: str,
) -> None:
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "kernels-community/flash-attn"},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.genrm_params.overwrite_output_dir,
        completion_only_loss=True,
        # Training parameters
        bf16=cfg.genrm_params.use_bf16,
        eval_strategy="no",
        eval_steps=None,
        gradient_accumulation_steps=cfg.genrm_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.genrm_params.learning_rate,
        lr_scheduler_type=cfg.genrm_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.genrm_params.lr_scheduler_kwargs,
        max_length=cfg.genrm_params.max_length,
        num_train_epochs=cfg.genrm_params.num_epochs,
        per_device_train_batch_size=cfg.genrm_params.batch_size,
        per_device_eval_batch_size=cfg.genrm_params.batch_size,
        seed=cfg.genrm_params.seed,
        warmup_ratio=cfg.genrm_params.warmup_ratio,
        weight_decay=cfg.genrm_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.genrm_params.logging_steps,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"CodeShield/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.genrm_params.save_steps,
        # Data parameters
        data_seed=cfg.genrm_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    trainer = SFTTrainer(model=model_name, args=config, train_dataset=data)
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def _count_tokens(prompt: Prompt, tokenizer) -> int:
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized_prompt = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
    return len(tokenized_prompt)


def _by_tokens(example, max_tokens: int) -> bool:
    return example["num_tokens"] <= max_tokens


def does_file_exist(file: Path) -> bool:
    return file.exists()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    model_short_name = cfg.genrm_params.model_path.split("/")[-1]
    wandb_run_name = f"genrm_{model_short_name}"
    output_dir = Path(f"{os.getenv('WORK')}/genrm_output/{wandb_run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "model.safetensors.index.json", "generation_config.json"]]):
        log.info(f"GenRM training files are already present in {output_dir}. Skipping.")
        return None

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.genrm_params.model_path)
    train_data = load_dataset(cfg.data.train)["train"]

    train_data = train_data.map(_create_training_dataset, num_proc=NUM_WORKERS, desc="Creating prompts")

    train_data = train_data.map(_count_tokens, fn_kwargs={"tokenizer": tokenizer}, num_proc=NUM_WORKERS, desc="Counting tokens")
    train_data = train_data.filter(_by_tokens, fn_kwargs={"max_tokens": cfg.genrm_params.max_length}, num_proc=NUM_WORKERS, desc="Filtering long sequences")

    log.info(f"Training {cfg.genrm_params.model_path} on {len(train_data)} examples")
    train_model(cfg, cfg.genrm_params.model_path, train_data, wandb_run_name, output_dir)
    log.info(f"Completed training {cfg.genrm_params.model_path} on {len(train_data)} examples")


if __name__ == "__main__":
    main()
