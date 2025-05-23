import logging
import os
import datasets
import polars as pl

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
log.setLevel(logging.INFO)


am = datasets.load_dataset("INSAIT-Institute/cold_start_dedupe_r1", num_proc=os.cpu_count())["train"]
rank1 = datasets.load_dataset("INSAIT-Institute/rank1_dedupe", num_proc=os.cpu_count())["train"]

am = pl.from_arrow(am.data.table)
rank1 = pl.from_arrow(rank1.data.table)
rank1 = rank1.cast({"verify_score": pl.Float64, "variability": pl.Float64})
rank1 = rank1.sample(fraction=1, shuffle=True)

am = am.with_columns(pl.col("question").rank("dense").alias("query_id"))

am_easy = am.filter(pl.col("variability") < 0.05)
am_hard = am.filter(pl.col("variability") >= 0.05)

# Sample only one correct answer per easy query
am_easy_single_ans = am_easy.group_by("query_id").agg(pl.all().sample(n=1, seed=42)).explode(pl.all().exclude("query_id"))

# Stage 1
stage1_code = am_easy_single_ans.filter(pl.col("category") == "code").sample(n=120_000, seed=42)
stage1_math = am_easy_single_ans.filter(pl.col("category") == "math").sample(n=80_000, seed=42)
stage1_rank1 = rank1.head(int(0.3 * len(rank1)))
stage1_rank1 = stage1_rank1.with_columns(pl.lit(0).alias("query_id").cast(pl.UInt32))
stage1 = pl.concat([stage1_code, stage1_math, stage1_rank1], how="align")
print(f"Stage 1 code: {len(stage1_code)}")
print(f"Stage 1 math: {len(stage1_math)}")
print(f"Stage 1 rank1: {len(stage1_rank1)}")

# Stage 2
am_easy_code_remaining = am_easy_single_ans.join(stage1, on="query_id", how="anti")
stage2_code = am_hard.filter(pl.col("category") == "code")
stage2_math = am_hard.filter(pl.col("category") == "math")
stage2_easy = am_easy_code_remaining.filter(pl.col("category").is_in(["code", "math"])).sample(n=40_000, seed=42)
stage2_rank1 = rank1.tail(int(0.7 * len(rank1)))
stage2_rank1 = stage2_rank1.with_columns(pl.lit(0).alias("query_id").cast(pl.UInt32))
stage2_other = am_hard.filter(pl.col("category").is_in(["other", "science", "instruction follow"])).sample(n=40_000, seed=42)

print(f"Stage 2 code: {len(stage2_code)}")
print(f"Stage 2 math: {len(stage2_math)}")
print(f"Stage 2 easy: {len(stage2_easy)}")
print(f"Stage 2 other: {len(stage2_other)}")
print(f"Stage 2 rank1: {len(stage2_rank1)}")
stage2 = pl.concat([stage2_code, stage2_math, stage2_easy, stage2_other, stage2_rank1], how="align")

stage1 = stage1.with_columns(pl.lit("stage_1").alias("currciulum_stage")).drop("query_id").sample(fraction=1, seed=42)
stage2 = stage2.with_columns(pl.lit("stage_2").alias("currciulum_stage")).drop("query_id").sample(fraction=1, seed=42)
final = pl.concat([stage1, stage2])
print("Final Dataset Statistics")
print(f"Total: {len(final)}")
print(f"Stage 1: {len(stage1)}")
print(f"Stage 2: {len(stage2)}")
print("Category distribution")
print(final.select(pl.col("category").value_counts()).unnest("category"))
data = datasets.DatasetDict({"train": datasets.Dataset(final.to_arrow())})
data.push_to_hub("CodeShield/coldstart_curriculum", private=True, max_shard_size="5GB")
