from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    path: str = Path(__file__).parent.parent.parent / "data/cold_start_data.parquet"


@dataclass
class SFTParams:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_short_name: str = model_name.split("/")[-1].lower()
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 2e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    logging_steps: int = 1000
    lr_scheduler_type: str = "cosine"
    gradient_accumulation_steps: int = 4
    output_dir: str = (Path(__file__).parent.parent / f"coldstart_output/{model_short_name}").as_posix()
    overwrite_output_dir: bool = True
    seed: int = 42
    use_bf16: bool = True
    hub_model_id: str = f"CodeShield/{model_short_name}-sft-coldstart"


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
