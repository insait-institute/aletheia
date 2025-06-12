import logging
import os
import re
from collections import Counter
from copy import deepcopy

# from datetime import timedelta
from pathlib import Path

import hydra
import numpy as np
import torch

# import torch.distributed as dist
from datasets import concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import wandb
from configs.grpo_config import Config

# dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=10))
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
wandb.login()
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-GRPO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
DAPO_ARGS = {"L_max": 7168, "L_cache": 1024}
PHI_ARGS = {"L_max": 8192, "L_pos_control": 6550, "L_neg_control": 820, "W_acc": 8 / 13, "W_rep": 1 / 13}
WANDB_NAMING_SCHEME = {
    "cfg.ablation_params.loss_type": {"bnpo": "b", "dr_grpo": "dr", "grpo": "g"},
    "kl": {"no": "n", "static": "s", "dynamic": "d"},
    "reward_type": {
        "correctness_only": "c",
        "correctness_format": "cf",
        "correctness_format_length": "cfl",
        "phi": "p",
    },
}


def wrap_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """
    copied from https://github.com/huggingface/trl/issues/2897
    Wraps a normal tokenizer to make sure its batch_decode does not skip special tokens
    """
    new_tokenizer = deepcopy(tokenizer)

    old_batch_decode = new_tokenizer.batch_decode

    def batch_decode(*args, **kwargs):
        """
        Batch decode sequences
        """
        kwargs["skip_special_tokens"] = False
        return old_batch_decode(*args, **kwargs)

    new_tokenizer.batch_decode = batch_decode
    return new_tokenizer


def correctness_reward(completions, verdict, **kwargs):
    contents = [completion[0]["content"].split("<solution>")[-1].split("</solution")[0].strip() for completion in completions]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, verdict)]


def format_reward(completions, **kwargs):
    pattern = r"^<reason>\n.*?\n</reason>\n<solution>\n(.*?)\n</solution>$"
    completion_contents = [completion[0]["content"].split("<|im_end|>")[0].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [0.0 if match and match.group(1) in ["Yes", "No"] else -1.0 for match in matches]


def length_reward(completion_ids, **kwargs):
    return [len(ids) / 8192 for ids in completion_ids]


def soft_overlong_punishment(completion_ids, **kwargs):
    # taken from https://github.com/huggingface/trl/issues/3130
    rewards = []
    for ids in completion_ids:
        completion_length = len(ids)
        if completion_length <= DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"]:
            rewards.append(0)
        elif DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"] < completion_length <= DAPO_ARGS["L_max"]:
            rewards.append((DAPO_ARGS["L_max"] - DAPO_ARGS["L_cache"] - completion_length) / DAPO_ARGS["L_cache"])
        else:
            rewards.append(-1)
    return rewards


def phi_reward(completions, completion_ids, verdict, **kwargs):
    rewards = []
    for completion, ids, ground_truth in zip(completions, completion_ids, verdict):
        L = len(ids)
        n = 5
        pattern = r"^<reason>\n.*?\n</reason>\n<solution>\n(.*?)\n</solution>$"
        completion_content = completion[0]["content"].split("<|im_end|>")[0].strip()
        match = re.match(pattern, completion_content, re.DOTALL)
        if "<|im_end|>" not in completion[0]["content"]:
            R_acc_scaled = -0.5
        elif not (match and match.group(1) in ["Yes", "No"]):
            R_acc_scaled = -1.0
        else:
            answer = completion_content.split("<solution>")[-1].split("</solution")[0].strip()
            if answer == ground_truth:
                rho = min(1, max(L - PHI_ARGS["L_pos_control"], 0) / (PHI_ARGS["L_max"] - PHI_ARGS["L_pos_control"]))
                R_max = 1.0
                R_min = 0.5
                R_acc_scaled = (R_min + 0.5 * (R_max - R_min) * np.cos(np.pi * rho)).item()
            else:
                rho = min(1, L / PHI_ARGS["L_neg_control"])
                R_max = -0.5
                R_min = -1.0
                R_acc_scaled = (R_min + 0.5 * (R_min - R_max) * np.cos(np.pi * rho)).item()

        n_grams = [tuple(ids[i : i + n]) for i in range(L - n + 1)]
        counts = Counter(n_grams)
        num_repeated_types = 0
        max_freq_of_repeated = 0
        for gram, freq in counts.items():
            if freq > 5:
                num_repeated_types += 1
                if freq > max_freq_of_repeated:
                    max_freq_of_repeated = freq

        total_types = len(counts)
        if num_repeated_types == 0:
            R_rep = 0.0
        else:
            ratio_type_repetition = num_repeated_types / total_types
            ratio_max_freq = max_freq_of_repeated / (L / 5)
            R_rep = -max(ratio_type_repetition, ratio_max_freq)

        rewards.append(PHI_ARGS["W_acc"] * R_acc_scaled + PHI_ARGS["W_rep"] * R_rep)

    return rewards


def filter_by_origin(example, origin):
    return example["origin"] == origin


@hydra.main(version_base=None, config_name="grpo_config")
def train(cfg: Config):
    model_short_name = cfg.grpo_params.model_name.split("/")[-1].lower().replace("sft-", "")
    wandb_run_name = (
        f"Loss-{WANDB_NAMING_SCHEME['cfg.ablation_params.loss_type'][cfg.ablation_params.loss_type]}_"
        + f"KL-{WANDB_NAMING_SCHEME['kl'][cfg.ablation_params.kl_penalty]}_"
        + f"Reward-{WANDB_NAMING_SCHEME['reward_type'][cfg.ablation_params.reward_type]}_"
        + f"Model-{model_short_name}"
    )
    output_dir = (Path(__file__).parent.parent / "grpo_output" / wandb_run_name).as_posix()

    if cfg.ablation_params.reward_type == "correctness_only":
        REWARD_FUNC = [correctness_reward]
    elif cfg.ablation_params.reward_type == "correctness_format":
        REWARD_FUNC = [correctness_reward, format_reward]
    elif cfg.ablation_params.reward_type == "correctness_format_length":
        REWARD_FUNC = [correctness_reward, format_reward, length_reward]
    elif cfg.ablation_params.reward_type == "phi":
        REWARD_FUNC = [phi_reward]
    else:
        raise ValueError(f"Unknown reward type: {cfg.ablation_params.reward_type}. Choose from 'correctness_only', 'correctness_format', 'correctness_format_length' or 'phi")

    if cfg.ablation_params.loss_type == "bnpo":
        REWARD_FUNC.append(soft_overlong_punishment)

    if cfg.ablation_params.kl_penalty == "dynamic" and cfg.ablation_params.ref_model_sync_steps <= 0:
        raise ValueError("kl_update_steps must be greater than 0 for dynamic KL penalty.")

    data = load_dataset(cfg.data.path)
    train_data, eval_data = data["train"], None

    # Training on a subset of the data for ablations
    target_size = int(0.05 * len(train_data))
    cpe_data = train_data.filter(filter_by_origin, fn_kwargs={"origin": 0}, num_proc=NUM_WORKERS, desc="Filtering CPE data")
    gp_data = train_data.filter(filter_by_origin, fn_kwargs={"origin": 1}, num_proc=NUM_WORKERS, desc="Filtering GP data")
    cpemd_data = train_data.filter(filter_by_origin, fn_kwargs={"origin": 2}, num_proc=NUM_WORKERS, desc="Filtering CPE_MD data")
    train_data = concatenate_datasets(
        [
            gp_data.shuffle(seed=cfg.grpo_params.seed).select(range(target_size // 2)),
            cpe_data.shuffle(seed=cfg.grpo_params.seed).select(range(target_size // 4)),
            cpemd_data.shuffle(seed=cfg.grpo_params.seed).select(range(target_size // 4)),
        ]
    ).shuffle(seed=cfg.grpo_params.seed)

    # if "test" in data:
    #     eval_data = data["test"]
    log.info(f"Loaded data from {cfg.data.path}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {NUM_WORKERS}")
    log.info(f"Number of GPUs: {os.environ.get('WORLD_SIZE', torch.cuda.device_count())}")

    # Updated model loading to use explicit BF16 tensor type
    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.bfloat16},
        # GRPO parameters
        beta=0.0 if cfg.ablation_params.kl_penalty == "no" else cfg.grpo_params.beta,
        epsilon=cfg.grpo_params.epsilon,
        epsilon_high=0.28 if cfg.ablation_params.loss_type == "bnpo" else None,
        learning_rate=cfg.grpo_params.learning_rate,
        loss_type=cfg.ablation_params.loss_type,
        mask_truncated_completions=True,
        sync_ref_model=cfg.ablation_params.kl_penalty == "dynamic",
        ref_model_mixup_alpha=cfg.ablation_params.ref_model_syncup_alpha,
        ref_model_sync_steps=cfg.ablation_params.ref_model_sync_steps,
        scale_rewards=not cfg.ablation_params.loss_type == "dr_grpo",
        # Training parameters
        bf16=cfg.grpo_params.use_bf16,
        bf16_full_eval=cfg.grpo_params.use_bf16,
        gradient_accumulation_steps=cfg.grpo_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=cfg.grpo_params.lr_scheduler_type,
        max_prompt_length=cfg.grpo_params.max_prompt_length,
        num_train_epochs=cfg.grpo_params.num_epochs,
        per_device_train_batch_size=cfg.grpo_params.batch_size,
        seed=cfg.grpo_params.seed,
        weight_decay=cfg.grpo_params.weight_decay,
        warmup_ratio=cfg.grpo_params.warmup_ratio,
        # Evaluation parameters
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.grpo_params.save_steps if eval_data else None,
        per_device_eval_batch_size=cfg.grpo_params.eval_batch_size if eval_data else None,
        # Checkpointing parameters
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.grpo_params.overwrite_output_dir,
        save_strategy="no",
        save_steps=cfg.grpo_params.save_steps,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.grpo_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Data parameters
        data_seed=cfg.grpo_params.seed,
        dataloader_num_workers=NUM_WORKERS,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        # Generation parameters
        max_completion_length=cfg.gen_params.max_completion_length,
        num_generations=cfg.gen_params.num_generations,
        temperature=cfg.gen_params.temperature,
        use_liger_loss=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
        vllm_tensor_parallel_size=cfg.gen_params.vllm_tensor_parallel_size,
        # Miscellaneous parameters
        ddp_backend="nccl",
        ddp_timeout=36000,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.grpo_params.model_name)
    tokenizer = wrap_tokenizer(tokenizer)  # Ensure batch_decode does not skip special tokens
    trainer = GRPOTrainer(
        model=cfg.grpo_params.model_name,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNC,
    )
    # Start training with explicit checkpoint resumption
    trainer.train()
    log.info("Training completed.")
    trainer.save_model(output_dir)
    # trainer.push_to_hub()
    # log.info(f"Model trained and saved to {output_dir}")


if __name__ == "__main__":
    train()
