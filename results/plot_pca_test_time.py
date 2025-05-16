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

        # Just use test_accuracy by default
        which = "test_accuracy"
        mean_accuracy = None
        std_accuracy = None

        # MCMC plots with interventions
        if "ablated_dataset" in df.columns:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]
            
            # [XXX]'s SAE stuff uses this column to indicate testing w/o intervention
            if matching[2] == 1:
                which = "test_accuracy_flipped"
        
        # MCMC plots w/o interventions
        else:
            intended_df = df

            if matching[2] == 1:
                which = "test_accuracy_flipped"       

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
    means = full_df["test_accuracy"].mean()
    std_devs = full_df["test_accuracy"].std()

    return means, std_devs


# %%
import pandas as pd
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt
import numpy as np

gender_base = pd.read_csv("gender_base.csv")
gender_pca = pd.read_csv("gender_pca.csv")

gender_base = gender_base[gender_base["random"] == False].copy()
gender_pca_interpreted = gender_pca[gender_pca["intervention"] == "interpreted"].copy()
gender_pca_test_time = gender_pca[gender_pca["intervention"] == "interpreted"].copy()

gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_pca_interpreted_means, gender_pca_interpreted_std_devs = get_gender_data(
    gender_pca_interpreted
)
gender_pca_test_time_means, gender_pca_test_time_std_devs = get_gender_data_test_time(
    gender_pca_test_time
)

mcmc_base = pd.read_csv("mcmc_base.csv")
mcmc_pca = pd.read_csv("mcmc_pca.csv")


mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(mcmc_base)

mcmc_pca_interpreted = mcmc_pca[mcmc_pca["intervention"] == "interpreted"].copy()
(
    mcmc_pca_interpreted_means,
    mcmc_pca_interpreted_std_devs,
    mcmc_pca_interpreted_labels,
) = get_mcmc_data(mcmc_pca_interpreted)

mcmc_pca_test_time = mcmc_pca[mcmc_pca["intervention"] == "interpreted"].copy()
mcmc_pca_test_time_means, mcmc_pca_test_time_std_devs, mcmc_pca_test_time_labels = get_mcmc_data_test_time(mcmc_pca_test_time)




# %%

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)
labels_sorted = [mcmc_pca_interpreted_labels[i] for i in sort_indices]

mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]

# Sort PCA data using the same indices
mcmc_pca_interpreted_means_sorted = [
    mcmc_pca_interpreted_means[i] for i in sort_indices
]
mcmc_pca_interpreted_std_devs_sorted = [
    mcmc_pca_interpreted_std_devs[i] for i in sort_indices
]

mcmc_pca_test_time_means_sorted = [
    mcmc_pca_test_time_means[i] for i in sort_indices
]
mcmc_pca_test_time_std_devs_sorted = [
    mcmc_pca_test_time_std_devs[i] for i in sort_indices
]

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.25  # Adjust width for three bars

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
    mcmc_pca_interpreted_means_sorted,
    width,
    yerr=mcmc_pca_interpreted_std_devs_sorted,
    capsize=5,
    label="Fine-tuning with Ablations",
)
rects3 = axs[0].bar(
    x + width,
    mcmc_pca_test_time_means_sorted,
    width,
    yerr=mcmc_pca_test_time_std_devs_sorted,
    capsize=5,
    label="Fine-tuning with Ablations + Test-time Ablation",
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
axs[0].tick_params(axis="y", labelsize=14)

# --- Gender Plotting ---
gender_labels = ["Gender Bias"]
x_gender = np.arange(len(gender_labels))

gender_width = 0.2

rects5 = axs[1].bar(
    x_gender - gender_width,
    [gender_base_means],
    gender_width,
    yerr=[gender_base_std_devs],
    capsize=5,
)
rects6 = axs[1].bar(
    x_gender,
    [gender_pca_interpreted_means],
    gender_width,
    yerr=[gender_pca_interpreted_std_devs],
    capsize=5,
)
rects7 = axs[1].bar(
    x_gender + gender_width,
    [gender_pca_test_time_means],
    gender_width,
    yerr=[gender_pca_test_time_std_devs],
    capsize=5,
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xlim(-0.75, 0.75)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].tick_params(axis="y", labelsize=14)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fontsize=15)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.95)
plt.show()
