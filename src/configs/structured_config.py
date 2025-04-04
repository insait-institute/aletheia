import os
from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class LocalModel:
    name: str = "Qwen/QwQ-32b"


@dataclass
class OpenAI:
    name: str = "deepseek-ai/DeepSeek-R1"
    service: str = "together"
    api_key: str = os.getenv(service.upper() + "_API_KEY")
    api_base: str = os.getenv(service.upper() + "_API_BASE")


@dataclass
class SParams:
    temperature: float = 0
    max_tokens: int = 10000
    n: int = 1
    seed: int = 42


@dataclass
class Data:
    name: str = "coseal/CodeUltraFeedback_binarized"
    split: str = "train"
    num_samples: int = 2
    seed: int = 42


@dataclass
class Retries:
    max_retries: int = 8
    retry_interval: int = 1
    max_retry_interval: int = 60
    retry_timeout: int = 60
    exponential_backoff: bool = True
    retry_when_blank: bool = False


@dataclass
class LocalConfig:
    model: LocalModel = field(default_factory=LocalModel)
    sparams: SParams = field(default_factory=SParams)
    data: Data = field(default_factory=Data)


@dataclass
class OpenAIConfig:
    model: OpenAI = field(default_factory=OpenAI)
    sparams: SParams = field(default_factory=SParams)
    data: Data = field(default_factory=Data)
    retries: Retries = field(default_factory=Retries)


defaults = [{"db": "check_rationales_local"}]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    db: Any = MISSING


cs = ConfigStore.instance()
cs.store(group="db", name="check_rationales_local", node=LocalConfig)
cs.store(group="db", name="check_rationales_openai", node=OpenAIConfig)
cs.store(name="config", node=Config)
