from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class LocalModel:
    name: str = "google/gemma-3-27b-it"


@dataclass
class SParams:
    temperature: float = 0
    max_tokens: int = 10000
    n: int = 1
    seed: int = 42


@dataclass
class Data:
    name: str = "CodeShield/Commit-Preference-Enhanced"
    chosen_col: str = "chosen"
    rejected_col: str = "rejected"


@dataclass
class MDConfig:
    model: LocalModel = field(default_factory=LocalModel)
    sparams: SParams = field(default_factory=SParams)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="markdownize_dataset_config", node=MDConfig)
