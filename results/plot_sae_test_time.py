# %%

import ast

def get_mcmc_data_test_time(full_df):
    assert "pair" in full_df.columns, "pair column not found"
    full_df["pair"] = full_df["pair"].apply(ast.literal_eval)

    # keep[2] is the index that is intended
    keep = [
        ("verbs", "sentiment", 1),
        ("sports", "pronouns", 0),
        ("pronouns", "sports", 0),
        ("sentiment", "verbs", 1),
        ("sentiment", "sports", 0),
        ("verbs", "sports", 0),
        ("sentiment", "pronouns", 0),
        ("verbs", "pronouns", 0),
    ]

    means = []
    std_devs = []
    labels = []

    # Loop through pairs and respective grouped df
    grouped_df = full_df.groupby("pair")
    for dataset_pair, df in grouped_df:

        # Find the matching pair in the keep list
        matching = None
        for pair in keep:
            if pair[0] == dataset_pair[0] and pair[1] == dataset_pair[1]:
                matching = pair
                break
        
        # If no matching pair, skip
        if matching is None:
            continue

        # Get the ablated dataset
        ablated_dataset = dataset_pair[1 if matching[2] == 0 else 0]

        # Just use test_intervention_accuracy by default
        which = "test_intervention_accuracy"
        mean_accuracy = None
        std_accuracy = None

        # MCMC plots with interventions
        if "ablated_dataset" in df.columns:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]
            
            # Helena's SAE stuff uses this column to indicate testing w/o intervention
            if matching[2] == 1:
                which = "test_intervention_accuracy_flipped"
        
        # MCMC plots w/o interventions
        else:
            intended_df = df

            if matching[2] == 1:
                which = "test_intervention_accuracy_flipped"       

        
        assert intended_df.shape[0] == 5

        mean_accuracy = intended_df[which].mean()
        std_accuracy = intended_df[which].std()

        means.append(mean_accuracy)
        std_devs.append(std_accuracy)

        if matching[2] == 1:
            labels.append(f"{dataset_pair[0]} | $\\mathbf{{{dataset_pair[1]}}}$")
        else:
            labels.append(f"$\\mathbf{{{dataset_pair[0]}}}$ | {dataset_pair[1]}")

    return means, std_devs, labels

def get_gender_data_test_time(full_df):
    means = full_df["test_intervention_accuracy"].mean()
    std_devs = full_df["test_intervention_accuracy"].std()

    return means, std_devs


# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt

gender_sae = pd.read_csv("gender_sae.csv")
gender_base = pd.read_csv("gender_base.csv")

mcmc_sae = pd.read_csv("mcmc_sae.csv")
mcmc_base = pd.read_csv("mcmc_base.csv")

gender_sae = gender_sae[gender_sae["random"] == False]
gender_base = gender_base[gender_base["random"] == False]

gender_sae_means, gender_sae_std_devs = get_gender_data(gender_sae)
gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_sae_means_test_time, gender_sae_std_devs_test_time = get_gender_data_test_time(gender_sae)

mcmc_sae_means, mcmc_sae_std_devs, mcmc_sae_labels = get_mcmc_data(mcmc_sae.copy())
mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(mcmc_base.copy())
mcmc_sae_means_test_time, mcmc_sae_std_devs_test_time, mcmc_sae_labels_test_time = get_mcmc_data_test_time(mcmc_sae.copy())

# %%
import numpy as np

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)
labels_sorted = [mcmc_sae_labels[i] for i in sort_indices]
mcmc_sae_means_sorted = [mcmc_sae_means[i] for i in sort_indices]
mcmc_sae_std_devs_sorted = [mcmc_sae_std_devs[i] for i in sort_indices]
mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]
mcmc_sae_means_test_time_sorted = [mcmc_sae_means_test_time[i] for i in sort_indices]
mcmc_sae_std_devs_test_time_sorted = [mcmc_sae_std_devs_test_time[i] for i in sort_indices]

# Sort PCA data using the same indices

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.2  # Adjust width for three bars

fig, axs = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [5, 1]}
)

# Plot MCMC data (grouped bars for each label, sorted)
rects1 = axs[0].bar(
    x - width,
    mcmc_base_means_sorted,
    width,
    yerr=mcmc_base_std_devs_sorted,
    capsize=5,
    label="No Intervention",
)
rects2 = axs[0].bar(
    x,
    mcmc_sae_means_sorted,
    width,
    yerr=mcmc_sae_std_devs_sorted,
    capsize=5,
    label="Fine-tuning with Ablations",
)
rects3 = axs[0].bar(
    x + width,
    mcmc_sae_means_test_time_sorted,
    width,
    yerr=mcmc_sae_std_devs_test_time_sorted,
    capsize=5,
    label="Fine-tuning with Ablations + Test-time Ablation",
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
# axs[0].legend(fontsize=14)
axs[0].tick_params(axis='y', labelsize=14)

# --- Gender Plotting ---
gender_labels = ["Gender Bias"]
x_gender = np.arange(len(gender_labels))

rects4 = axs[1].bar(
    x_gender - width,
    [gender_base_means],
    width,
    yerr=[gender_base_std_devs],
    capsize=5,
)
rects5 = axs[1].bar(
    x_gender,
    [gender_sae_means],
    width,
    yerr=[gender_sae_std_devs],
    capsize=5,
)
rects6 = axs[1].bar(
    x_gender + width,
    [gender_sae_means_test_time],
    width,
    yerr=[gender_sae_std_devs_test_time],
    capsize=5,
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xlim(-1,1)
axs[1].set_xticks(x_gender)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].tick_params(axis='y', labelsize=14)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fontsize=15)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.95)
plt.show()
