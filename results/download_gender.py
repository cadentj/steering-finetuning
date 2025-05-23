# %%

import wandb
import pandas as pd
from tqdm import tqdm

ENTITY = "steering-finetuning"
PROJECT = "gender"
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
    "run_name" : "pair",
    "config.seed" : "seed",
    "summaryMetrics.test/acc" : "test_accuracy",
    "summaryMetrics.train/accuracy" : "train_accuracy",
    "summaryMetrics.no_interventions/acc" : "no_interventions_accuracy",
    "summaryMetrics.no_interventions/acc_flipped" : "no_interventions_accuracy_flipped",
}

# Only keep the columns in keep_and_rename_map, then rename them
filtered_df = df[list(keep_and_rename_map.keys())].rename(columns=keep_and_rename_map)

# Drop baseline runs
filtered_df = filtered_df[filtered_df['no_interventions_accuracy'].notna()]

# Drop rows where 'no_intervention' is in the 'pair' column
filtered_df = filtered_df[~filtered_df['pair'].str.contains('no_intervention')]

# %%

def make_intervention_col(row):
    pair_name = row['pair']
    if "top_intervention" in pair_name:
        row['intervention'] = "top_intervention"
    elif "random_intervention" in pair_name:
        row['intervention'] = "random_intervention"
    else:
        row['intervention'] = "none"
    return row

filtered_df = filtered_df.apply(make_intervention_col, axis=1)

filtered_df.to_csv("gender_pca.csv", index=False)