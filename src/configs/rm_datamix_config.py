from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    name: str = "CodeShield/CPE_Markdownized"
    seed: int = 42
    _lambda: float = 0.33


@dataclass
class Config:
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="rm_datamix_config", node=Config)
