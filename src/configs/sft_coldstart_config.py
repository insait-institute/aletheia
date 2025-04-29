from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    path: str = "wetsoledrysoul/cold_start_data"
    split_ratio: float = 0.9


@dataclass
class SFTParams:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    logging_steps: int = 1
    max_length: int = 8192
    lr_scheduler_type: str = "linear"
    gradient_accumulation_steps: int = 1
    overwrite_output_dir: bool = True
    seed: int = 42
    use_bf16: bool = True
    resume_training_if_possible: bool = True
    deepspeed_config_path: str = "configs/deepspeed_config.json"


@dataclass
class WandbParams:
    run_name: str = "sft-coldstart-test-run"
    log_level: str = "info"


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    sft_params: SFTParams = field(default_factory=SFTParams)
    wandb_params: WandbParams = field(default_factory=WandbParams)


cs = ConfigStore.instance()
cs.store(name="sft_coldstart_config", node=Config)
