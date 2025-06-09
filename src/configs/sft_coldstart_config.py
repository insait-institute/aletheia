from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    path: str = "CodeShield/coldstart_curriculum_v2"
    split_ratio: float = 1.0


@dataclass
class SFTParams:
    batch_size: int = 4
    deepspeed_config_path: str = "configs/deepspeed_config.json"
    gradient_accumulation_steps: int = 16
    learning_rate: float = 5e-5
    logging_steps: int = 1
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr": 5e-6})
    lr_scheduler_type: str = "cosine_with_min_lr"
    max_length: int = 6200
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    num_epochs: int = 1
    overwrite_output_dir: bool = True
    resume_training_if_possible: bool = True
    save_steps: float = 0.25
    seed: int = 42
    tp_size: int = 0
    use_bf16: bool = True
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-2


@dataclass
class WandbParams:
    log_level: str = "info"
    run_name: str = "sft-coldstart-test-run"


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    sft_params: SFTParams = field(default_factory=SFTParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="sft_coldstart_config", node=Config)
