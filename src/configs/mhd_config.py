import os
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class Basics:
    scratch_dir: str = f"/work/{os.getenv('USER')}"
    output_base: str = "dedup"
    logs_folder: str = "SlurmLogs/dedup_logs"
    local_logs_folder: str = "SlurmLogs/dedup_local_logs"
    total_tasks: int = 8


@dataclass
class Data:
    model_name: str = "coderpile"
    dataset_prefix: str = "wetsoledrysoul/CoderPile-Annot-Set"
    dataset_name: str = "wetsoledrysoul/CoderPile-Annot-Set"
    text_key: str = "prompt"


@dataclass
class Slurm:
    nodelist: str = "msp3-7"
    exclude_nodes: str = "msp3-1"


@dataclass
class Minhash:
    precision: int = 64
    num_buckets: int = 32
    hashes_per_bucket: int = 4
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
