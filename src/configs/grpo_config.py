from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    path: str = "CodeShield/RLData_RM"
    split_ratio: float = 1.0


@dataclass
class GRPOParams:
    batch_size: int = 32
    beta: float = 1e-3
    epsilon: float = 0.2
    epsilon_high: float = 0.4
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-6
    logging_steps: int = 1
    # lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    # lr_scheduler_type: str = "cosine_with_min_lr"
    max_prompt_length: int = 6200
    model_name: str = "CodeShield/sft-qwen3-0.6b"
    num_epochs: int = 1
    overwrite_output_dir: bool = True
    resume_training_if_possible: bool = True
    save_steps: float = 0.1
    seed: int = 42
    use_bf16: bool = True
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-2


@dataclass
class GenParams:
    max_completion_length: int = 8192
    num_generations: int = 8
    temperature: float = 0.8
    vllm_dtype: str = "bfloat16"
    vllm_gpu_memory_utilization: float = 0.4
    vllm_server_host: str = "localhost"
    vllm_server_port: int = 8000
    vllm_server_timeout: int = 600
    vllm_tensor_parallel_size: int = 4


@dataclass
class WandbParams:
    log_level: str = "info"


@dataclass
class AblationParams:
    reward_type: str = "correctness_only"  # Options: "correctness_only", "correctness_format", "correctness_format_length", "phi"
    kl_penalty: str = "no"  # Options: "no", "static", "dynamic"
    ref_model_syncup_alpha: float = 1.0  # 1.0 is a hard reset of the reference model, 0.0 is no syncup
    ref_model_sync_steps: int = 100  # Number of steps to update KL penalty. Only relevant if kl_penalty is "dynamic"
    loss_type: str = "dr_grpo"  # Options: "dr_grpo", "grpo", "dnpo"


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    gen_params: GenParams = field(default_factory=GenParams)
    grpo_params: GRPOParams = field(default_factory=GRPOParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)
    ablation_params: AblationParams = field(default_factory=AblationParams)


cs = ConfigStore.instance()
cs.store(name="grpo_config", node=Config)
