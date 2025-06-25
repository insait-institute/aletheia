from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Inference:
    model_name: str = "CodeShield/sft-qwen3-1.7b"
    N: int = 16
    output_dir: str = "outputs/goldilocks"
    shard: int = 0
    total_shards: int = 4


@dataclass
class Config:
    inference: Inference = field(default_factory=Inference)


cs = ConfigStore.instance()
cs.store(name="goldilocks_config", node=Config)
