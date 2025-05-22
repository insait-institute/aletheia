import os
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Basics:
    scratch_dir: str = f"/scratch/{os.getenv('USER')}"
    output_base: str = "coldstart_data_dedup"
    logs_folder: str = "SlurmLogs/dedup_logs"
    local_logs_folder: str = "SlurmLogs/dedup_local_logs"
    total_tasks: int = 32


@dataclass
class Data:
    model_name: str = "rank1"
    dataset_prefix: str = "CodeShield/rank1_dedupe"
    dataset_name: str = "jhu-clsp/rank1-training-data"
    text_key: str = "messages"


@dataclass
class Slurm:
    nodelist: str = "msp3-7"
    exclude_nodes: str = "gcp-eu-1,gcp-eu-2,gcpl4-eu-0,gcpl4-eu-1,gcpl4-eu-2,gcpl4-eu-3,gcpl4-eu-4,gcpl4-eu-5,gcpl4-eu-6,gcpl4-eu-7"


@dataclass
class Minhash:
    precision: int = 64
    num_buckets: int = 32
    hashes_per_bucket: int = 8
    n_grams: int = 5
    seed: int = 42


@dataclass
class Config:
    basics: Basics = field(default_factory=Basics)
    data: Data = field(default_factory=Data)
    slurm: Slurm = field(default_factory=Slurm)
    minhash: Minhash = field(default_factory=Minhash)


cs = ConfigStore.instance()
cs.store(name="mhd_config", node=Config)
