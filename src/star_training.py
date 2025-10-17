import logging
import os
import pickle
from pathlib import Path
from typing import List

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from cerebrm_prompts import STAR_PROMPT
from cerebrm_rewards import extract_boxed_contents_list
from configs.schema import Config
from utils import Prompt, get_generated_text, maybe_resume_training, run_inference

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-STAR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_prompts(example, cfg: Config, hinted=False):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    if hinted:
        candidate_str += f"\n\nHint: The correct answer is {example['chosen_answer']}"
    example["prompt"] = [
        {
            "role": "user",
            "content": STAR_PROMPT.format(
                question=example["query"],
                candidates=candidate_str,
                valid_options=", ".join(potential_answers),
            ).strip(),
        },
    ]
    return example


def generate_completions(cfg: Config, prompts: List[Prompt], model: str) -> List[str]:
    # Sampling K responses from the current model
    responses = run_inference(prompts, model, temperature=0.0, max_tokens=8192, n=1, dp_size=torch.cuda.device_count(), tp_size=1, max_model_len=12288)
    completions = get_generated_text(responses)
    return [x[0] for x in completions]


def score_completions(completions: List[str], ground_truths: List[str]) -> List[bool]:
    # Scoring the K responses using a verifiable reward
    model_answers = [extract_boxed_contents_list(x) for x in completions]
    scored_completions = [model_ans == gt for model_ans, gt in zip(model_answers, ground_truths)]
    return scored_completions


def train_star_model(
    cfg: Config,
    model_path_episode: str,
    stage3_data: Dataset,
    wandb_run_name: str,
    output_dir: str,
) -> None:
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "kernels-community/flash-attn"},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.star_params.overwrite_output_dir,
        completion_only_loss=True,
        # Training parameters
        bf16=cfg.star_params.use_bf16,
        eval_strategy="no",
        eval_steps=None,
        gradient_accumulation_steps=cfg.star_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.star_params.learning_rate,
        lr_scheduler_type=cfg.star_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.star_params.lr_scheduler_kwargs,
        max_length=cfg.star_params.max_length,
        num_train_epochs=cfg.star_params.num_epochs,
        per_device_train_batch_size=cfg.star_params.batch_size,
        per_device_eval_batch_size=cfg.star_params.batch_size,
        seed=cfg.star_params.seed,
        warmup_ratio=cfg.star_params.warmup_ratio,
        weight_decay=cfg.star_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.star_params.logging_steps,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"wetsoledrysoul/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.star_params.save_steps,
        # Data parameters
        data_seed=cfg.star_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    trainer = SFTTrainer(model=model_path_episode, args=config, train_dataset=stage3_data)
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def _count_tokens(example, tokenizer):
    prompt = example["prompt"] + example["completion"]
    prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized_prompt = tokenizer(prompt, padding=False, truncation=False)["input_ids"]
    example["num_tokens"] = len(tokenized_prompt)
    return example


def create_messages_from_completions(prompts, completions):
    processed_prompts, processed_completions = [], []
    for prompt, completion in zip(prompts, completions):
        if prompt[-1]["role"] == "assistant":
            completion = prompt[-1]["content"] + completion
            prompt = prompt[:-1]
        processed_prompts.append(prompt)
        processed_completions.append([{"role": "assistant", "content": completion}])
    return processed_prompts, processed_completions


def _by_tokens(example, max_tokens: int) -> bool:
    return example["num_tokens"] <= max_tokens


def does_file_exist(file: Path) -> bool:
    return file.exists()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    ### Sanity checks
    assert cfg.star_stage in [1, 2, 3], "Invalid star_stage. Must be 1 for generating and scoring completions, 2 for generating and scoring hinted completions, or 3 for training."
    assert cfg.star_episode >= 0, "Invalid star_episode. Must be a non-negative integer."
    model_short_name = cfg.star_params.model_path.split("/")[-1]
    wandb_run_name = f"star_{model_short_name}_ep{cfg.star_episode}"
    model_path_episode = cfg.star_params.model_path if cfg.star_episode == 0 else f"wetsoledrysoul/star_{model_short_name}_ep{cfg.star_episode - 1}"
    output_dir = Path(f"{os.getenv('WORK')}/star_output/{wandb_run_name}")
    if cfg.star_stage == 2:
        if not all([does_file_exist(output_dir / x) for x in ["stage1_correct_prompts.pkl", "stage1_correct_completions.pkl", "stage1_incorrect_indices.pkl"]]):
            raise ValueError(f"Stage 1 files are missing in {output_dir}. Please run stage 1 for episode {cfg.star_episode} before proceeding.")
    if cfg.star_stage == 3:
        if not all([does_file_exist(output_dir / x) for x in ["correct_prompts.pkl", "correct_completions.pkl"]]):
            raise ValueError(f"Stage 2 files are missing in {output_dir}. Please run stage 2 for episode {cfg.star_episode} before proceeding.")
    ### End sanity checks

    ### Check if stage has been completed previously
    if cfg.star_stage == 1:
        if all([does_file_exist(output_dir / x) for x in ["stage1_correct_prompts.pkl", "stage1_correct_completions.pkl", "stage1_incorrect_indices.pkl"]]):
            log.info(f"Stage 1 files are already present in {output_dir}. Skipping.")
            return None
    elif cfg.star_stage == 2:
        if all([does_file_exist(output_dir / x) for x in ["correct_prompts.pkl", "correct_completions.pkl"]]):
            log.info(f"Stage 2 files are already present in {output_dir}. Skipping.")
            return None
    elif cfg.star_stage == 3:
        if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "model.safetensors.index.json", "generation_config.json"]]):
            log.info(f"Stage 3 files are already present in {output_dir}. Skipping.")
            return None
    ### End Checks

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.star_params.model_path)
    train_data = load_dataset(cfg.data.train)["train"]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = train_data.map(_create_prompts, fn_kwargs={"cfg": cfg}, num_proc=NUM_WORKERS, desc="Creating prompts")
    unhinted_prompts = list(train_data["prompt"])

    log.info(f"Starting star episode {cfg.star_episode}")
    if cfg.star_stage == 1:
        log.info(f"Processing {len(unhinted_prompts)} prompts for stage 1 with {model_path_episode}")
        stage1_completions = generate_completions(cfg, unhinted_prompts, model_path_episode)
        log.info(f"Scoring {len(stage1_completions)} prompts for stage 1")
        scored_completions = score_completions(stage1_completions, list(train_data["chosen_answer"]))
        log.info(f"Number of correct responses in stage 1: {sum(scored_completions)} ({(sum(scored_completions) / len(scored_completions)) * 100:.2f}%)")

        correct_indices = [i for i, score in enumerate(scored_completions) if score]
        incorrect_indices = [i for i, score in enumerate(scored_completions) if not score]

        stage1_correct_prompts, stage1_correct_completions = create_messages_from_completions(
            [x for i, x in enumerate(unhinted_prompts) if i in correct_indices], [x for i, x in enumerate(stage1_completions) if i in correct_indices]
        )

        with open(output_dir / "stage1_correct_prompts.pkl", "wb") as f:
            pickle.dump(stage1_correct_prompts, f)
        with open(output_dir / "stage1_correct_completions.pkl", "wb") as f:
            pickle.dump(stage1_correct_completions, f)
        with open(output_dir / "stage1_incorrect_indices.pkl", "wb") as f:
            pickle.dump(incorrect_indices, f)
        log.info(f"Completed Stage 1 of star episode {cfg.star_episode}")

    elif cfg.star_stage == 2:
        train_data = train_data.map(_create_prompts, fn_kwargs={"cfg": cfg, "hinted": True}, num_proc=NUM_WORKERS, desc="Creating prompts")

        with open(output_dir / "stage1_incorrect_indices.pkl", "rb") as f:
            stage2_indices = pickle.load(f)
        with open(output_dir / "stage1_correct_prompts.pkl", "rb") as f:
            stage1_correct_prompts = pickle.load(f)
        with open(output_dir / "stage1_correct_completions.pkl", "rb") as f:
            stage1_correct_completions = pickle.load(f)

        stage2_prompts = list(train_data.select(stage2_indices)["prompt"])
        stage2_answers = list(train_data.select(stage2_indices)["chosen_answer"])

        log.info(f"Processing {len(stage2_prompts)} prompts for stage 2")
        stage2_completions = generate_completions(cfg, stage2_prompts, model_path_episode)
        log.info(f"Scoring {len(stage2_completions)} prompts for stage 2")
        scored_completions = score_completions(stage2_completions, stage2_answers)
        log.info(f"Number of correct responses in stage 2: {sum(scored_completions)} ({(sum(scored_completions) / len(scored_completions)) * 100:.2f}%)")

        correct_indices = [i for i, score in zip(stage2_indices, scored_completions) if score]
        stage2_correct_prompts = [x for i, x in enumerate(unhinted_prompts) if i in correct_indices]
        stage2_correct_completions = [x for x, score in zip(stage2_completions, scored_completions) if score]

        stage2_correct_prompts, stage2_correct_completions = create_messages_from_completions(stage2_correct_prompts, stage2_correct_completions)
        correct_prompts = stage1_correct_prompts + stage2_correct_prompts
        correct_completions = stage1_correct_completions + stage2_correct_completions
        with open(output_dir / "correct_prompts.pkl", "wb") as f:
            pickle.dump(correct_prompts, f)
        with open(output_dir / "correct_completions.pkl", "wb") as f:
            pickle.dump(correct_completions, f)
        log.info(f"Completed Stage 2 of star episode {cfg.star_episode}")
    else:
        correct_prompts, correct_completions = [], []
        with open(output_dir / "correct_prompts.pkl", "rb") as f:
            correct_prompts.extend(pickle.load(f))
        with open(output_dir / "correct_completions.pkl", "rb") as f:
            correct_completions.extend(pickle.load(f))
        stage3_data = Dataset.from_dict({"prompt": correct_prompts, "completion": correct_completions})
        stage3_data = stage3_data.map(_count_tokens, fn_kwargs={"tokenizer": tokenizer}, num_proc=NUM_WORKERS, desc="Counting tokens")
        stage3_data = stage3_data.filter(_by_tokens, fn_kwargs={"max_tokens": cfg.star_params.max_length}, num_proc=NUM_WORKERS, desc="Filtering long sequences")

        log.info(f"Training {cfg.star_params.model_path} on {len(stage3_data)} examples in stage three for episode {cfg.star_episode}")
        train_star_model(cfg, cfg.star_params.model_path, stage3_data, wandb_run_name, output_dir)
        log.info(f"Completed Stage 3 of star episode {cfg.star_episode}")


if __name__ == "__main__":
    """
    STaR stages: 
    Stage 1: Sample 1 response from the current model for each prompt in the training set and score all the responses
    Stage 2: Sample and score responses with prompts augmented with a hint for the incorrect prompts in stage 1
    Stage 3: Train on the final dataset of correct responses from stage 1 and 2
    """
    main()
