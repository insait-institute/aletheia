import os
from pathlib import Path

from datasets import load_dataset

# for lang in ["ruby"]:
#     exec_1 = load_dataset("wetsoledrysoul/" + lang + "_execs")["train"]
#     exec_2 = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_execs*.parquet")["train"]

#     exec_1 = exec_1.to_pandas()
#     exec_2 = exec_2.to_pandas()
#     # compute pass_rates
#     exec_1["pass_rate"] = exec_1["num_passed"] / (exec_1["num_passed"] + exec_1["num_failed"])
#     exec_2["pass_rate"] = exec_2["num_passed"] / (exec_2["num_passed"] + exec_2["num_failed"])
#     exec_1 = exec_1.drop(["num_passed", "num_failed", "final_verdict"], axis=1).explode(["completions", "pass_rate"])
#     exec_2 = exec_2.drop(["num_passed", "num_failed", "final_verdict"], axis=1).explode(["completions", "pass_rate"])

#     df = exec_2.copy()
#     df["old_pass_rate"] = exec_1["pass_rate"].copy()
#     print("len df", len(df))
#     df = df[df["pass_rate"] == df["old_pass_rate"]].drop(["old_pass_rate"], axis=1)
#     print("len df after", len(df))
#     df = df.groupby(["id"]).agg({"prompt_id": "first", "language": "first", "generator": "first", "completions": list, "pass_rate": list}).reset_index()
#     Dataset.from_pandas(df).to_parquet(f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_verified.parquet")

# for lang in ["javascript", "ruby", "rust", "csharp", "d"]:
#     exec = load_dataset("wetsoledrysoul/" + lang + "_execs")["train"]
#     # exec = load_dataset("parquet", data_files=f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_veryfied.parquet")["train"]
#     exec = exec.to_pandas()
#     exec["pairable"] = exec["pass_rate"].apply(lambda x: not all([y == 1 for y in x]) and any([y == 1 for y in x]))
#     exec = exec[exec["pairable"]].reset_index(drop=True)
#     indices = exec["id"].unique().tolist()
#     with open(f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_indices.txt", "w") as f:
#         f.write("\n".join(indices))

for lang in ["ruby"]:
    exec_1 = load_dataset("parquet", data_files=[f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_execs2_filtered_{i}_of_10.parquet" for i in range(4)])["train"]
    exec_2 = load_dataset("parquet", data_files=[f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_execs_filtered_{i}_of_10.parquet" for i in range(4)])["train"]
    indices = Path(f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_indices.txt").read_text().split("\n")
    exec_1 = exec_1.to_pandas()
    exec_2 = exec_2.to_pandas()
    exec_1 = exec_1[exec_1["id"].isin(indices)].reset_index(drop=True).drop_duplicates(subset="id", keep="first")
    exec_2 = exec_2[exec_2["id"].isin(indices)].reset_index(drop=True).drop_duplicates(subset="id", keep="first")
    breakpoint()
    # compute pass_rates
    exec_1["pass_rate"] = exec_1["num_passed"] / (exec_1["num_passed"] + exec_1["num_failed"])
    exec_2["pass_rate"] = exec_2["num_passed"] / (exec_2["num_passed"] + exec_2["num_failed"])
    exec_1 = exec_1.drop(["num_passed", "num_failed", "final_verdict"], axis=1).explode(["completions", "pass_rate"])
    exec_2 = exec_2.drop(["num_passed", "num_failed", "final_verdict"], axis=1).explode(["completions", "pass_rate"])
    df = exec_2.copy()
    df["old_pass_rate"] = exec_1["pass_rate"].copy()
    print("len df", len(df))
    new_df = df[df["pass_rate"] == df["old_pass_rate"]].drop(["old_pass_rate"], axis=1)
    print("len df after", len(new_df))
    new_df = new_df.groupby(["id"]).agg({"prompt_id": "first", "language": "first", "generator": "first", "completions": list, "pass_rate": list}).reset_index()
    breakpoint()
    # Dataset.from_pandas(df).to_parquet(f"{os.getenv('HOME')}/SandboxFusion/outputs/{lang}_veryveryfied.parquet")
