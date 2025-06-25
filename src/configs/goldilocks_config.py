from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Inference:
    model_name: str = "CodeShield/sft-qwen3-1.7b"
    N: int = 16
    output_dir: str = "outputs/goldilocks"
    shard: int = 0
    total_shards: int = 4
    goldilocks_type: str = "all"  # all (selects 70k enhanced, 50k commit preference and all genpref) or commitpref (selects all commitpref)


@dataclass
class Config:
    inference: Inference = field(default_factory=Inference)


cs = ConfigStore.instance()
cs.store(name="goldilocks_config", node=Config)
