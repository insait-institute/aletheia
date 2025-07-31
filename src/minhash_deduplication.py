import os

import hydra
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.writers.huggingface import ParquetWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.logging import logger
from datatrove.utils.typeshelper import Languages

from configs.mhd_config import Config


def _dict_to_str_adapter(self, data: dict, path: str, id_in_file: int | str):
    """
    The default data adapter to adapt input data into the datatrove Document format

    Args:
        data: a dictionary with the "raw" representation of the data
        path: file path or source for this sample
        id_in_file: its id in this particular file or source

    Returns: a dictionary with text, id, media and metadata fields

    """
    metadata = data.pop("metadata", {})
    if isinstance(metadata, str):
        import json

        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            pass
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}
    text = data.get(self.text_key, "")
    if isinstance(text, list) and isinstance(text[0], dict):
        assert text[1]["role"] == "user", "Expected the second dict to be a user prompt"
        text = text[1]["content"]  # assuming the first dict is a system prompt
    return {
        "text": text,
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": metadata | data,  # remaining data goes into metadata
    }


@hydra.main(version_base=None, config_name="mhd_config")
def main(cfg: Config):
    # you can also change ngrams or the number of buckets and their size here
    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=cfg.minhash.precision),
        n_grams=cfg.minhash.n_grams,
        num_buckets=cfg.minhash.num_buckets,
        hashes_per_bucket=cfg.minhash.hashes_per_bucket,
        seed=cfg.minhash.seed,
    )  # better precision -> fewer false positives (collisions)

    SCRATCH_DIR = cfg.basics.scratch_dir
    DATA_NAME = cfg.data.dataset_name if cfg.data.dataset_name else f"{cfg.data.dataset_prefix}_{cfg.data.model_name}"
    LOGS_FOLDER = f"{SCRATCH_DIR}/{cfg.basics.logs_folder}_{cfg.data.model_name}"
    LOCAL_LOGS_FOLDER = f"{SCRATCH_DIR}/{cfg.basics.local_logs_folder}_{cfg.data.model_name}"
    OUTPUT_DIR = f"{SCRATCH_DIR}/{cfg.basics.output_base}_{cfg.data.model_name}"
    MINHASH_BASE_PATH = f"{OUTPUT_DIR}/minhash"
    DEDUP_OUTPUT_PATH = f"{OUTPUT_DIR}/deduped_output"
    REMOVED_OUTPUT_PATH = f"{OUTPUT_DIR}/removed_instances"

    TOTAL_TASKS = cfg.basics.total_tasks

    INPUT_READER = HuggingFaceDatasetReader(DATA_NAME, adapter=_dict_to_str_adapter, text_key=cfg.data.text_key, dataset_options={"split": "train"}, doc_progress=True)

    os.makedirs(MINHASH_BASE_PATH, exist_ok=True)
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    os.makedirs(LOCAL_LOGS_FOLDER, exist_ok=True)
    os.makedirs(DEDUP_OUTPUT_PATH, exist_ok=True)
    os.makedirs(REMOVED_OUTPUT_PATH, exist_ok=True)

    # Stage 1: Signature generation - maximize parallelism
    stage1 = SlurmPipelineExecutor(
        job_name="mh1",
        pipeline=[
            INPUT_READER,
            MinhashDedupSignature(output_folder=f"{MINHASH_BASE_PATH}/signatures", config=minhash_config, language=Languages.english),
        ],
        tasks=TOTAL_TASKS,
        time="24:00:00",
        partition="batch",
        logging_dir=f"{LOGS_FOLDER}/signatures",
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
        mem_per_cpu_gb=3,
        cpus_per_task=2,
        workers=8,
        sbatch_args={
            "nodelist": cfg.slurm.nodelist,
            "exclude": cfg.slurm.exclude_nodes,
        },
    )

    # Stage 2: Bucket processing
    stage2 = SlurmPipelineExecutor(
        job_name="mh2",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{MINHASH_BASE_PATH}/signatures",
                output_folder=f"{MINHASH_BASE_PATH}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,  # 14
        time="4:00:00",
        partition="batch",
        logging_dir=f"{LOGS_FOLDER}/buckets",
        depends=stage1,
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/buckets/slurm_logs",
        workers=8,
        mem_per_cpu_gb=2,
        cpus_per_task=2,
        sbatch_args={
            "nodelist": cfg.slurm.nodelist,
            "exclude": cfg.slurm.exclude_nodes,
        },
    )

    # Stage 3: Clustering
    stage3 = SlurmPipelineExecutor(
        job_name="mh3",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{MINHASH_BASE_PATH}/buckets",
                output_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        time="4:00:00",
        partition="batch",
        logging_dir=f"{LOGS_FOLDER}/clusters",
        mem_per_cpu_gb=35,
        cpus_per_task=2,
        depends=stage2,
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/clusters/slurm_logs",
        sbatch_args={
            "nodelist": cfg.slurm.nodelist,
            "exclude": cfg.slurm.exclude_nodes,
        },
    )

    # Stage 4: Filtering - same configuration as stage 1
    stage4 = SlurmPipelineExecutor(
        job_name="mh4",
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=f"{MINHASH_BASE_PATH}/remove_ids",
                exclusion_writer=ParquetWriter(
                    output_folder=REMOVED_OUTPUT_PATH,
                    max_file_size=4_500_000_000,  # ~4.5GB
                    expand_metadata=True,
                ),
            ),
            ParquetWriter(
                output_folder=DEDUP_OUTPUT_PATH,
                max_file_size=4_500_000_000,  # ~4.5GB
                expand_metadata=True,
            ),
        ],
        tasks=TOTAL_TASKS,
        time="4:00:00",
        partition="batch",
        logging_dir=f"{LOGS_FOLDER}/filter",
        depends=stage3,
        slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filter/slurm_logs",
        mem_per_cpu_gb=8,
        workers=8,
        cpus_per_task=2,
        sbatch_args={
            "nodelist": cfg.slurm.nodelist,
            "exclude": cfg.slurm.exclude_nodes,
        },
    )

    stage4.run()
    logger.info("All deduplication stages have been submitted.")


if __name__ == "__main__":
    main()
