from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    path: str = "CodeShield/GenRM_data_0.33"
    split_ratio: float = 1.0


@dataclass
class GRPOParams:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    beta: float = 0.04
    epsilon: float = 0.2
    loss_type: str = "dr_grpo"
    batch_size: int = 4
    num_epochs: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    logging_steps: int = 1
    max_prompt_length: int = 4096
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 1e-5})
    gradient_accumulation_steps: int = 16
    overwrite_output_dir: bool = True
    seed: int = 42
    use_bf16: bool = True
    resume_training_if_possible: bool = True
    deepspeed_config_path: str = "configs/deepspeed_config.json"


@dataclass
class GenParams:
    temperature: float = 0.5
    max_completion_length: int = 8192
    num_generations: int = 8
    vllm_server_host: str = "0.0.0.0"
    vllm_server_port: int = 8000
    vllm_server_timeout: int = 600
    vllm_gpu_memory_utilization: float = 0.95
    vllm_dtype: str = "bfloat16"
    vllm_max_model_len: Optional[int] = None


@dataclass
class WandbParams:
    run_name: str = "grpo-test-run"
    log_level: str = "info"


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    grpo_params: GRPOParams = field(default_factory=GRPOParams)
    gen_params: GenParams = field(default_factory=GenParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="grpo_config", node=Config)
