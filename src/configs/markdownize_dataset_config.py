from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class LocalModel:
    name: str = "google/gemma-3-27b-it"
    max_model_len: int = 8192


@dataclass
class SParams:
    temperature: float = 0
    max_tokens: int = 8192
    n: int = 1
    seed: int = 42


@dataclass
class Data:
    name: str = "CodeShield/Commit-Preference-Enhanced"
    col: str = "rejected"
    split: str = "test"


@dataclass
class MDConfig:
    model: LocalModel = field(default_factory=LocalModel)
    sparams: SParams = field(default_factory=SParams)
    data: Data = field(default_factory=Data)
    mode: str = "markdownize"


cs = ConfigStore.instance()
cs.store(name="markdownize_dataset_config", node=MDConfig)
