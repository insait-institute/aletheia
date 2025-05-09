import os
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Defaults:
    scratch_dir: str = f"/scratch/{os.getenv('USER')}"
    output_base: str = "coldstart_data_dedup"
    logs_folder = "SlurmLogs/dedup_logs"
    local_logs_folder = "SlurmLogs/dedup_local_logs"
    total_tasks: int = 32


class Data:
    model_name: str = "7b"
    dataset_prefix = "CodeShield/cold_start_data_predupe"
    text_key: str = "messages"


class Slurm:
    nodelist: str = "gcpcpu-eu-1"
    exclude_nodes: list = ["gcp-eu-1,gcp-eu-2,gcpl4-eu-0,gcpl4-eu-1,gcpl4-eu-2,gcpl4-eu-3,gcpl4-eu-4,gcpl4-eu-5,gcpl4-eu-6,gcpl4-eu-7"]


class Minhash:
    precision: int = 64
    num_buckets: int = 32
    hashes_per_bucket: int = 8
    n_grams: int = 5
    seed: int = 42


@dataclass
class Config:
    defaults: Defaults = field(default_factory=Defaults)
    data: Data = field(default_factory=Data)
    slurm: Slurm = field(default_factory=Slurm)
    minhash: Minhash = field(default_factory=Minhash)


cs = ConfigStore.instance()
cs.store(name="mhd_config", node=Config)
