from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Data:
    path: str = "CodeShield/RLData_RM"
    origin: str = "cpe_md"
    subset: float = 0.25


@dataclass
class Model:
    name: str = "CodeShield/sft-qwen3-0.6b"
    N: int = 8


@dataclass
class Output:
    dir: str = "outputs/evaluation_results"


@dataclass
class Config:
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    output: Output = field(default_factory=Output)


cs = ConfigStore.instance()
cs.store(name="evaluate_config", node=Config)
