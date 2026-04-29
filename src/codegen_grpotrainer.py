"""GRPO trainer for code generation that uses a trained pairwise code verifier
as the reward signal in place of executable test cases.

The verifier was trained on a list-wise selection task (pick the best of N
candidate codes). To use it as a per-rollout reward inside GRPO, we follow the
Bootstrapped Relative Policy Optimization (BRPO) recipe from Writing-Zero
(https://arxiv.org/pdf/2506.00103):

    For each prompt, sample G policy rollouts {o_1, ..., o_G}.
    Pick one rollout o_ref uniformly at random as the temporary reference.
    For each i != ref, query the verifier as a 2-candidate listwise comparison
    (the natural N=2 reduction of its training task) and set
        R_i = +1 if o_i is preferred over o_ref else -1.
    R_ref is fixed at 0. Advantages are A_i = R_i directly (no group
    normalization). Position bias is suppressed by voting on both candidate
    orderings; a candidate only earns +1 if it wins every voting condition.
    Groups whose preference distribution is too skewed
    (|sum R_i| / G > tau_filter) are dropped from the gradient update.
"""

import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import torch
from configs.schema import Config as BaseConfig
from datasets import load_dataset
from kernels import has_kernel
from omegaconf import OmegaConf
from openai import OpenAI
from prompts import LIST_REWARD_PROMPT
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from utils import maybe_resume_training

import wandb

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
wandb.login()
os.environ["WANDB_ENTITY"] = "Aletheia"
os.environ["WANDB_PROJECT"] = "GRPO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1

CODE_BLOCK_PATTERN = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
BOXED_VERDICT_PATTERN = re.compile(r"\\boxed\{([AB])\}")


@dataclass
class VerifierParams:
    base_url: str = "http://localhost:8001/v1"
    model: str = None
    api_key: str = "EMPTY"
    voting: int = 1
    max_tokens: int = 4096
    temperature: float = 0.6
    brpo_filter_threshold: float = 0.6
    default_language: str = "python"
    prompt_key: str = "query"
    language_key: str = "language"


@dataclass
class Config(BaseConfig):
    verifier_params: Optional[VerifierParams] = field(default_factory=VerifierParams)


def extract_code(text: str) -> str:
    """Return the last fenced code block, or the raw text if none was emitted."""
    if not text:
        return ""
    blocks = CODE_BLOCK_PATTERN.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


def build_pairwise_judge_messages(question: str, code_a: str, code_b: str, language: str) -> list:
    """Render the (N=2) listwise prompt the verifier was trained on."""
    candidates = f"[CANDIDATE_A]\n```{language}\n{code_a}\n```\n[/CANDIDATE_A]\n\n[CANDIDATE_B]\n```{language}\n{code_b}\n```\n[/CANDIDATE_B]"
    content = LIST_REWARD_PROMPT.format(
        question=question,
        candidates=candidates,
        valid_options="A, B",
    ).strip()
    return [{"role": "user", "content": content}]


def parse_judge_verdict(text: str) -> Optional[str]:
    """Return 'A' or 'B' from the verifier's response (post-thought), else None."""
    if not text:
        return None
    payload = text.split("</think>")[-1]
    match = BOXED_VERDICT_PATTERN.search(payload)
    return match.group(1) if match else None


def create_codegen_prompts(example, prompt_key="query"):
    example["prompt"] = [{"role": "user", "content": example[prompt_key].strip()}]
    return example


class CodegenGRPOTrainer(GRPOTrainer):
    """GRPO trainer wired to a frozen pairwise code verifier with BRPO advantages.

    Compatible with the standard ``trl.GRPOTrainer`` lifecycle. The verifier is
    expected to be served behind an OpenAI-compatible HTTP endpoint (e.g. via
    ``vllm serve``) so it can run on a dedicated GPU while the policy keeps its
    own colocated vLLM engine.
    """

    def __init__(
        self,
        *args,
        verifier_base_url: str,
        verifier_model: str,
        verifier_api_key: str = "EMPTY",
        verifier_voting: int = 1,
        verifier_max_tokens: int = 4096,
        verifier_temperature: float = 0.6,
        brpo_filter_threshold: float = 0.6,
        default_language: str = "python",
        language_key: str = "language",
        prompt_key: str = "query",
        **kwargs,
    ):
        # super() requires at least one reward function. The verifier-driven
        # advantages bypass it (returning zero contributions), so we hand it a
        # null function that keeps gathering/logging machinery happy.
        kwargs.setdefault("reward_funcs", [self._null_reward_func])
        super().__init__(*args, **kwargs)
        # Long timeout: a single batched call covers the whole local step's
        # judge prompts, and large groups (G * voting * 2) can take a while.
        self.verifier_client = OpenAI(base_url=verifier_base_url, api_key=verifier_api_key, timeout=600.0)
        self.verifier_model = verifier_model
        self.verifier_voting = max(1, int(verifier_voting))
        self.verifier_max_tokens = verifier_max_tokens
        self.verifier_temperature = verifier_temperature
        self.brpo_filter_threshold = brpo_filter_threshold
        self.default_language = default_language
        self.language_key = language_key
        self.prompt_key = prompt_key
        # Verifier tokenizer applies its chat template client-side so we can
        # drop down to /v1/completions and submit every judge prompt in one
        # batched HTTP request (vLLM continuous-batches them server-side).
        self.verifier_tokenizer = AutoTokenizer.from_pretrained(verifier_model)

    @staticmethod
    def _null_reward_func(prompts, completions, **kwargs):
        return [0.0] * len(completions)

    def _render_judge_prompt(self, question: str, code_a: str, code_b: str, language: str) -> str:
        messages = build_pairwise_judge_messages(question, code_a, code_b, language)
        return self.verifier_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _compute_brpo_advantages(
        self,
        queries: List[str],
        languages: List[str],
        completions_text: List[str],
    ) -> torch.Tensor:
        device = self.accelerator.device
        G = self.num_generations
        n = len(completions_text)
        assert n % G == 0, f"num_completions={n} not divisible by num_generations={G}"
        n_groups = n // G

        codes = [extract_code(t) for t in completions_text]
        # For every (group, candidate, voting_round, A/B ordering) emit one
        # judge prompt. A candidate only earns +1 if it wins all 2 * voting
        # conditions, so we expand to that fan-out up front and parse back.
        judge_prompts: List[str] = []
        spec_meta: List[tuple] = []  # (group, candidate, ordering)  ordering: "AB" | "BA"
        for g in range(n_groups):
            start = g * G
            ref_idx = random.randrange(G)
            ref_code = codes[start + ref_idx]
            question = queries[start + ref_idx]
            language = languages[start + ref_idx] or self.default_language
            for i in range(G):
                if i == ref_idx:
                    continue
                cand_code = codes[start + i]
                prompt_ab = self._render_judge_prompt(question, cand_code, ref_code, language)
                prompt_ba = self._render_judge_prompt(question, ref_code, cand_code, language)
                for _ in range(self.verifier_voting):
                    judge_prompts.append(prompt_ab)
                    spec_meta.append((g, i, "AB"))
                    judge_prompts.append(prompt_ba)
                    spec_meta.append((g, i, "BA"))

        verdicts = self._batched_judge(judge_prompts)

        # Aggregate: a candidate wins a condition if the verifier picks it
        # (i.e. "A" under AB ordering, "B" under BA ordering). Require all
        # 2 * voting conditions to be wins for R_i = +1.
        rounds_per_cand = self.verifier_voting * 2
        win_counts: dict = {}
        for (g, i, ordering), verdict in zip(spec_meta, verdicts):
            cand_wins = (verdict == "A" and ordering == "AB") or (verdict == "B" and ordering == "BA")
            win_counts[(g, i)] = win_counts.get((g, i), 0) + (1 if cand_wins else 0)

        advantages = torch.zeros(n, dtype=torch.float32, device=device)
        for (g, i), wins in win_counts.items():
            advantages[g * G + i] = 1.0 if wins == rounds_per_cand else -1.0
        # Reference rollouts (one per group) keep advantage = 0.

        adv_grp = advantages.view(n_groups, G)
        skew = adv_grp.sum(dim=1).abs() / G
        keep = (skew <= self.brpo_filter_threshold).float().unsqueeze(1)
        adv_grp = adv_grp * keep

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["brpo/groups_dropped_frac"].append(1.0 - keep.mean().item())
        self._metrics[mode]["brpo/mean_preference"].append(adv_grp.mean().item())
        self._metrics[mode]["brpo/mean_abs_skew"].append(skew.mean().item())

        return adv_grp.view(-1)

    def _batched_judge(self, judge_prompts: List[str]) -> List[Optional[str]]:
        """Send every judge prompt in a single /v1/completions call so vLLM
        can continuous-batch them, returning per-prompt verdicts ('A'/'B'/None).
        """
        if not judge_prompts:
            return []
        try:
            resp = self.verifier_client.completions.create(
                model=self.verifier_model,
                prompt=judge_prompts,
                max_tokens=self.verifier_max_tokens,
                temperature=self.verifier_temperature,
                n=1,
            )
        except Exception as exc:
            log.warning(f"Batched verifier call failed; treating as no-preference: {exc}")
            return [None] * len(judge_prompts)
        # OpenAI returns one Choice per prompt with `index` matching the input
        # ordering; reorder defensively in case the server doesn't preserve it.
        verdicts: List[Optional[str]] = [None] * len(judge_prompts)
        for choice in resp.choices:
            if 0 <= choice.index < len(verdicts):
                verdicts[choice.index] = parse_judge_verdict(choice.text)
        return verdicts

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)

        # ``inputs`` is already expanded to (per_device_batch * num_generations)
        # by the repeat sampler, so positions align 1:1 with completion_ids.
        completions_text = self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        queries = [ex[self.prompt_key] for ex in inputs]
        languages = [ex.get(self.language_key, self.default_language) for ex in inputs]

        output["advantages"] = self._compute_brpo_advantages(queries, languages, completions_text)
        return output


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg: Config):
    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    model_short_name = cfg.grpo_params.model_path.split("/")[-1].lower()
    wandb_run_name = f"CodegenGRPO-{model_short_name}-brpo-so{cfg.grpo_params.generate_every}"
    output_dir = f"{os.getenv('WORK')}/grpo_output/{wandb_run_name}"

    if cfg.grpo_params.kl_penalty == "dynamic" and cfg.grpo_params.ref_model_sync_steps <= 0:
        raise ValueError("ref_model_sync_steps must be greater than 0 for dynamic KL penalty.")

    if cfg.data.train.endswith(".parquet"):
        train_data = load_dataset("parquet", data_files=cfg.data.train)["train"]
    else:
        train_data = load_dataset(cfg.data.train)["train"]
    train_data = train_data.map(
        create_codegen_prompts,
        fn_kwargs={"prompt_key": cfg.verifier_params.prompt_key},
        num_proc=NUM_WORKERS,
        desc="Creating prompts",
    )

    eval_data = None
    if cfg.data.val:
        if cfg.data.val.endswith(".parquet"):
            eval_data = load_dataset("parquet", data_files=cfg.data.val)["train"]
        else:
            eval_data = load_dataset(cfg.data.val)["train"]
        eval_data = eval_data.map(
            create_codegen_prompts,
            fn_kwargs={"prompt_key": cfg.verifier_params.prompt_key},
            num_proc=NUM_WORKERS,
            desc="Creating prompts",
        )

    log.info(f"Example prompt: {train_data['prompt'][0]}")
    log.info(f"Loaded data from {cfg.data.train}")
    log.info(f"Train size: {len(train_data)}")
    log.info(f"Eval size: {len(eval_data) if eval_data else 'N/A'}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Number of CPUs: {NUM_WORKERS}")
    log.info(f"Number of GPUs: {os.environ.get('WORLD_SIZE', torch.cuda.device_count())}")

    kernel = None
    if has_kernel("kernels-community/flash-attn3"):
        kernel = "kernels-community/flash-attn3"
    elif has_kernel("kernels-community/flash-attn2"):
        kernel = "kernels-community/flash-attn2"
    elif has_kernel("kernels-community/flash-attn"):
        kernel = "kernels-community/flash-attn"
    if kernel:
        log.info(f"Using attention kernel: {kernel}")

    config = GRPOConfig(
        model_init_kwargs={"attn_implementation": kernel},
        beta=0.0 if cfg.grpo_params.kl_penalty == "no" else cfg.grpo_params.beta,
        epsilon=cfg.grpo_params.epsilon,
        epsilon_high=cfg.grpo_params.epsilon_high,
        learning_rate=cfg.grpo_params.learning_rate,
        loss_type=cfg.grpo_params.loss_type,
        mask_truncated_completions=True,
        sync_ref_model=cfg.grpo_params.kl_penalty == "dynamic",
        ref_model_mixup_alpha=cfg.grpo_params.ref_model_mixup_alpha,
        ref_model_sync_steps=cfg.grpo_params.ref_model_sync_steps,
        bf16=cfg.grpo_params.use_bf16,
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
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.grpo_params.save_steps if eval_data else None,
        per_device_eval_batch_size=cfg.grpo_params.eval_batch_size if eval_data else None,
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.grpo_params.overwrite_output_dir,
        save_strategy="steps",
        save_steps=cfg.grpo_params.save_steps,
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.grpo_params.logging_steps,
        load_best_model_at_end=True if eval_data else False,
        report_to="wandb",
        run_name=wandb_run_name,
        data_seed=cfg.grpo_params.seed,
        dataloader_num_workers=NUM_WORKERS,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        max_completion_length=cfg.gen_params.max_completion_length,
        num_generations=cfg.gen_params.num_generations,
        temperature=cfg.gen_params.temperature,
        use_liger_loss=False,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_server_host=cfg.gen_params.vllm_server_host,
        vllm_server_port=cfg.gen_params.vllm_server_port,
        vllm_server_timeout=cfg.gen_params.vllm_server_timeout,
        vllm_tensor_parallel_size=cfg.gen_params.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=cfg.gen_params.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=True,
        steps_per_generation=cfg.grpo_params.gradient_accumulation_steps * cfg.grpo_params.generate_every,
        importance_sampling_level=cfg.grpo_params.importance_sampling_level,
        # We overwrite advantages directly with BRPO values; the parent's
        # mean/std normalization runs on the dummy reward so it does not
        # interact with the final advantages.
        scale_rewards="none",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.grpo_params.model_path)
    trainer = CodegenGRPOTrainer(
        model=cfg.grpo_params.model_path,
        args=config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        verifier_base_url=cfg.verifier_params.base_url,
        verifier_model=cfg.verifier_params.model,
        verifier_api_key=cfg.verifier_params.api_key,
        verifier_voting=cfg.verifier_params.voting,
        verifier_max_tokens=cfg.verifier_params.max_tokens,
        verifier_temperature=cfg.verifier_params.temperature,
        brpo_filter_threshold=cfg.verifier_params.brpo_filter_threshold,
        default_language=cfg.verifier_params.default_language,
        language_key=cfg.verifier_params.language_key,
        prompt_key=cfg.verifier_params.prompt_key,
    )
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
