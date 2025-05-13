from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class OpenAI:
    name: str = "deepseek-ai/DeepSeek-R1"
    service: str = "openrouter"


@dataclass
class SParams:
    max_tokens: int = 16192
    seed: int = 42
    chunk_batch_size: int = 100


@dataclass
class Data:
    name: str = "wetsoledrysoul/CPE_Markdownized"
    train_split: str = "train"
    test_split: str = "test"
    seed: int = 42
    chosen_col: str = "chosen_md"
    rejected_col: str = "rejected_md"
    input_col: str = "input"


@dataclass
class Config:
    model: OpenAI = field(default_factory=OpenAI)
    sparams: SParams = field(default_factory=SParams)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="r1_sdg_config", node=Config)
