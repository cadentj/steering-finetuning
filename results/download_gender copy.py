# %%

import wandb
import pandas as pd
from tqdm import tqdm

ENTITY = "steering-finetuning"
PROJECT = "gender_sae"
CSV_PATH = "wandb_run_stats.csv"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}")

run_stats = []
for run in tqdm(runs):
    stats = {
        "run_id": run.id,
        "run_name": run.name,
        **run.summary,
        **run.config,
        **run._attrs  # includes info like state, created_at, etc.
    }
    run_stats.append(stats)

# Normalize and save to CSV
stats_df = pd.json_normalize(run_stats)
stats_df.to_csv(CSV_PATH, index=False)
print(f"Saved run stats to {CSV_PATH}")

# %%

import pandas as pd

# Read the CSV file
df = pd.read_csv("wandb_run_stats.csv")

keep_and_rename_map = {
    "run_name" : "name",
    "config.seed" : "seed",
    "summaryMetrics.test/acc" : "test_accuracy",
    "summaryMetrics.deployed/acc" : "deployed_accuracy",
    "summaryMetrics.deployed/acc_flipped" : "deployed_accuracy_flipped",
}

# Only keep the columns in keep_and_rename_map, then rename them
filtered_df = df[list(keep_and_rename_map.keys())].rename(columns=keep_and_rename_map)

# %%

def make_intervention_col(row):
    name = row['name']
    if "top_intervention" in name:
        row['intervention'] = "top"
    elif "random_intervention" in name:
        row['intervention'] = "random"
    elif "test_only" in name:
        row['intervention'] = "test_only"
    elif "autointerp" in name:
        row['intervention'] = "autointerp"
    elif "intervention" in name:
        row['intervention'] = "interpreted"
    else:
        row['intervention'] = "base"
    return row

filtered_df = filtered_df.apply(make_intervention_col, axis=1)

filtered_df.to_csv("gender_sae_autointerp.csv", index=False)

# %%

filtered_df