from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Model:
    name: str = "Qwen/QwQ-32b"


@dataclass
class SParams:
    temperature: float = 0.8
    max_tokens: int = 1000


@dataclass
class Data:
    name: str = "coseal/CodeUltraFeedback_binarized"
    split: str = "train"
    num_samples: int = 250
    seed: int = 42


@dataclass
class Config:
    model: Model = field(default_factory=Model)
    sparams: SParams = field(default_factory=SParams)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="check_rationales", node=Config)
