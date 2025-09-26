from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Data:
    train: str = None
    val: str = None


@dataclass
class GRPOParams:
    beta: float = None
    epsilon: float = None
    epsilon_high: float = None
    eval_batch_size: int = None
    learning_rate: float = None
    logging_steps: int = None
    loss_type: str = None
    lr_scheduler_type: str = None
    max_prompt_length: int = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    save_steps: float = None
    save_total_limit: int = None
    seed: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    kl_penalty: str = None
    ref_model_sync_steps: int = None
    ref_model_mixup_alpha: float = None
    model_path: str = None
    batch_size: int = None
    gradient_accumulation_steps: int = None
    importance_sampling_level: str = None
    generate_every: int = None
    scale_rewards: str = None


@dataclass
class GenParams:
    max_completion_length: int = None
    num_generations: int = None
    temperature: float = None
    vllm_dtype: str = None
    vllm_gpu_memory_utilization: float = None
    vllm_server_host: str = None
    vllm_server_port: int = None
    vllm_server_timeout: int = None
    vllm_tensor_parallel_size: int = None


@dataclass
class SFTParams:
    batch_size: int = None
    deepspeed_config_path: str = None
    gradient_accumulation_steps: int = None
    learning_rate: float = None
    logging_steps: int = None
    lr_scheduler_kwargs: dict = None
    lr_scheduler_type: str = None
    max_length: int = None
    model_name: str = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    resume_training_if_possible: bool = None
    save_steps: float = None
    save_total_limit: int = None
    seed: int = None
    tp_size: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None


@dataclass
class RAFTParams:
    batch_size: int = None
    deepspeed_config_path: str = None
    gradient_accumulation_steps: int = None
    learning_rate: float = None
    logging_steps: int = None
    lr_scheduler_kwargs: dict = None
    lr_scheduler_type: str = None
    max_length: int = None
    model_name: str = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    resume_training_if_possible: bool = None
    save_steps: float = None
    save_total_limit: int = None
    seed: int = None
    tp_size: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    episode_num: int = None
    num_generations: int = None
    thinking_model: bool = None


@dataclass
class STaRParams:
    batch_size: int = None
    deepspeed_config_path: str = None
    gradient_accumulation_steps: int = None
    learning_rate: float = None
    logging_steps: int = None
    lr_scheduler_kwargs: dict = None
    lr_scheduler_type: str = None
    max_length: int = None
    model_name: str = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    resume_training_if_possible: bool = None
    save_steps: float = None
    save_total_limit: int = None
    seed: int = None
    tp_size: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    episode_num: int = None


@dataclass
class WandbParams:
    log_level: str = None


@dataclass
class Config:
    data: Optional[Data] = field(default_factory=Data)
    gen_params: Optional[GenParams] = field(default_factory=GenParams)
    grpo_params: Optional[GRPOParams] = field(default_factory=GRPOParams)
    sft_params: Optional[SFTParams] = field(default_factory=SFTParams)
    raft_params: Optional[RAFTParams] = field(default_factory=RAFTParams)
    star_params: Optional[STaRParams] = field(default_factory=STaRParams)
    wandb_params: Optional[WandbParams] = field(default_factory=WandbParams)
    reward_type: Optional[str] = None
