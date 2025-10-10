import logging
import os
import pickle
import random
from pathlib import Path
from typing import List

import hydra
import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from cerebrm_prompts import RAFT_PROMPT_NOTHINK, RAFT_PROMPT_THINK
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
os.environ["WANDB_PROJECT"] = "CerebRM-RAFT"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_prompts(example, cfg: Config):
    potential_answers = ["[[A]]", "[[B]]", "[[C]]", "[[D]]", "[[E]]"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i[2]}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i[2]}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    RAFT_PROMPT = RAFT_PROMPT_THINK if cfg.raft_params.thinking_model else RAFT_PROMPT_NOTHINK
    example["prompt"] = [
        {
            "role": "user",
            "content": RAFT_PROMPT.format(
                question=example["query"],
                candidates=candidate_str,
                valid_options=", ".join(potential_answers),
            ).strip(),
        },
    ]
    if cfg.raft_params.thinking_model:
        example["prompt"].append(
            {"role": "assistant", "content": "<think>\n"},
        )  # to avoid bug in reward calculation in older trl versions
    return example


def generate_completions(cfg: Config, prompts: List[Prompt], model: str) -> List[List[str]]:
    # Sampling K responses from the current model
    responses = run_inference(
        prompts,
        model,
        temperature=1.0,
        max_tokens=cfg.raft_params.max_length - 4096,
        n=cfg.raft_params.num_generations,
        dp_size=torch.cuda.device_count(),
        tp_size=1,
        max_model_len=cfg.raft_params.max_length,
    )
    completions = get_generated_text(responses)
    return completions


def score_completions(completions: List[List[str]], ground_truths: List[str]) -> List[List[bool]]:
    # Scoring the K responses using a verifiable reward
    model_answers = [[extract_boxed_contents_list(y) for y in x] for x in completions]
    scored_completions = [[model_ans == gt for model_ans in model_ans_list] for model_ans_list, gt in zip(model_answers, ground_truths)]
    return scored_completions


def train_raft_model(
    cfg: Config,
    model_path_episode: str,
    stage3_data: Dataset,
    wandb_run_name: str,
    output_dir: str,
    eval_data: Dataset | None,
) -> None:
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": "kernels-community/flash-attn"},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.raft_params.overwrite_output_dir,
        completion_only_loss=True,
        # Training parameters
        bf16=cfg.raft_params.use_bf16,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.raft_params.save_steps if eval_data else None,
        gradient_accumulation_steps=cfg.raft_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.raft_params.learning_rate,
        lr_scheduler_type=cfg.raft_params.lr_scheduler_type,
        lr_scheduler_kwargs=cfg.raft_params.lr_scheduler_kwargs,
        max_length=cfg.raft_params.max_length,
        num_train_epochs=cfg.raft_params.num_epochs,
        per_device_train_batch_size=cfg.raft_params.batch_size,
        per_device_eval_batch_size=cfg.raft_params.batch_size,
        seed=cfg.raft_params.seed,
        warmup_ratio=cfg.raft_params.warmup_ratio,
        weight_decay=cfg.raft_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.raft_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"CodeShield/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.raft_params.save_steps,
        save_total_limit=cfg.raft_params.save_total_limit,
        # Data parameters
        data_seed=cfg.raft_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    trainer = SFTTrainer(model=model_path_episode, args=config, train_dataset=stage3_data)
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


def construct_data_for_training(stage1_prompts, episode_completions, scored_completions, tokenizer) -> Dataset:
    prompts, completions, num_tokens = [], [], []
    for prompt, completions_list, scores in zip(stage1_prompts, episode_completions, scored_completions):
        completions_list = [c for c, s in zip(completions_list, scores) if s]
        if not completions_list:
            continue
        completion = random.sample(completions_list, 1)[0]
        if prompt[-1]["role"] == "assistant":
            completion = prompt[-1]["content"] + completion
            prompt = prompt[:-1]
        completion = [{"role": "assistant", "content": completion}]
        prompts.append(prompt)
        completions.append(completion)
    # print(type(prompts), type(completions), type(prompts[0]), type(completions[0]), prompts[0], completions[0])
    num_tokens = [_count_tokens(prompt + completion, tokenizer) for prompt, completion in zip(prompts, completions)]
    return Dataset.from_dict({"prompt": prompts, "completion": completions, "num_tokens": num_tokens})


def does_file_exist(file: Path) -> bool:
    return file.exists()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    ### Sanity checks
    assert cfg.raft_stage in [1, 2], "Invalid raft_stage. Must be 1 for generating and scoring completions, or 2 for training."
    assert cfg.raft_episode >= 0, "Invalid raft_episode. Must be a non-negative integer."
    model_short_name = cfg.raft_params.model_path.split("/")[-1]
    wandb_run_name = f"RAFT_{model_short_name}_ep{cfg.raft_episode}"
    model_path_episode = cfg.raft_params.model_path if cfg.raft_episode == 0 else f"CodeShield/{wandb_run_name}"
    output_dir = Path(f"{os.getenv('WORK')}/raft_output/{wandb_run_name}")
    if cfg.raft_stage == 2:
        if not all([does_file_exist(output_dir / x) for x in ["completions.pkl", "scored_completions.pkl"]]):
            raise ValueError(f"Stage 1 files are missing in {output_dir}. Please run stage 1 for episode {cfg.raft_episode} before proceeding.")
    ### End sanity checks
    ### Check if stage has been completed previously
    if cfg.raft_stage == 1:
        if all([does_file_exist(output_dir / x) for x in ["completions.pkl", "scored_completions.pkl"]]):
            log.info(f"Stage 1 files are already present in {output_dir}. Skipping.")
            return None
    elif cfg.raft_stage == 2:
        if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "model.safetensors.index.json", "generation_config.json"]]):
            log.info(f"Stage 2 files are already present in {output_dir}. Skipping.")
            return None
    ### End Checks

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.raft_params.model_path)
    train_data = load_dataset(cfg.data.train)["train"]
    train_data = train_data.shuffle(seed=cfg.raft_params.seed).select(range(5120))
    if cfg.data.val:
        eval_data = load_dataset(cfg.data.val)
        if "test_weak_easy" in eval_data:
            eval_data = eval_data["test_weak_easy"]
        elif "Full" in eval_data:
            eval_data = eval_data["Full"]
        else:
            eval_data = eval_data["test"]
    else:
        eval_data = None

    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = train_data.map(_create_prompts, fn_kwargs={"cfg": cfg}, num_proc=NUM_WORKERS, desc="Creating prompts")
    stage1_prompts = list(train_data["prompt"])
    log.info(f"Starting RAFT episode {cfg.raft_episode}")
    if cfg.raft_stage == 1:
        log.info(f"Processing {len(stage1_prompts)} prompts for stage one")
        episode_completions = generate_completions(cfg, stage1_prompts, model_path_episode)
        log.info(f"Scoring {len(episode_completions)} prompts for stage two")
        scored_completions = score_completions(episode_completions, list(train_data["chosen_answer"]))
        with open(output_dir / "completions.pkl", "wb") as f:
            pickle.dump(episode_completions, f)
        with open(output_dir / "scored_completions.pkl", "wb") as f:
            pickle.dump(scored_completions, f)
        log.info(f"Completed Stage 1 of RAFT episode {cfg.raft_episode}")

    else:
        with open(output_dir / "completions.pkl", "rb") as f:
            episode_completions = pickle.load(f)
        with open(output_dir / "scored_completions.pkl", "rb") as f:
            scored_completions = pickle.load(f)
        stage3_data = construct_data_for_training(stage1_prompts, episode_completions, scored_completions, tokenizer)
        log.info(f"Training {model_path_episode} on {len(stage3_data)} examples in Stage 2")
        train_raft_model(cfg, model_path_episode, stage3_data, wandb_run_name, output_dir, eval_data)
        log.info(f"Completed Stage 2 of RAFT episode {cfg.raft_episode}")


if __name__ == "__main__":
    """
    Usage guide:
    Since only one vllm object can be created at a time (without running into vllm errors), the script is designed to be run in two `sub_stages` per RAFT episode.
    For each episode, run the following commands sequentially:
    
    raft_training.py +raft_stage=1 raft_episode=0
    raft_training.py +raft_stage=2 raft_episode=0
    raft_training.py +raft_stage=1 raft_episode=1
    raft_training.py +raft_stage=2 raft_episode=1
    
    and so on.
    """
    main()
