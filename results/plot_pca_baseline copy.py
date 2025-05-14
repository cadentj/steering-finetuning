# %%

import pandas as pd
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt
import numpy as np

gender_base = pd.read_csv("gender_base.csv")
gender_pca = pd.read_csv("gender_pca.csv")

gender_sae_top = pd.read_csv("gender_sae_top.csv")

gender_base = gender_base[gender_base["random"] == False]
gender_interpreted_pca = gender_pca[gender_pca["intervention"] == "interpreted"]
gender_top_pca = gender_pca[gender_pca["intervention"] == "top"]

# %%

gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_pca_means, gender_pca_std_devs = get_gender_data(gender_interpreted_pca)
gender_top_pca_means, gender_top_pca_std_devs = get_gender_data(gender_top_pca)

gender_sae_top_means, gender_sae_top_std_devs = get_gender_data(gender_sae_top)

mcmc_base = pd.read_csv("mcmc_base.csv")
mcmc_pca = pd.read_csv("mcmc_pca.csv")
mcmc_sae_top = pd.read_csv("mcmc_sae_top.csv")

mcmc_pca_interpreted = mcmc_pca[
    mcmc_pca["intervention"] == "interpreted"
].copy()
mcmc_pca_top_intervention = mcmc_pca[mcmc_pca["intervention"] == "top"].copy()

mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(
    mcmc_base.copy()
)
(
    mcmc_pca_interpreted_means,
    mcmc_pca_interpreted_std_devs,
    mcmc_pca_interpreted_labels,
) = get_mcmc_data(mcmc_pca_interpreted)
(
    mcmc_pca_top_intervention_means,
    mcmc_pca_top_intervention_std_devs,
    mcmc_pca_top_intervention_labels,
) = get_mcmc_data(mcmc_pca_top_intervention)


# %%

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)[::-1]
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

mcmc_pca_top_intervention_means_sorted = [
    mcmc_pca_top_intervention_means[i] for i in sort_indices
]
mcmc_pca_top_intervention_std_devs_sorted = [
    mcmc_pca_top_intervention_std_devs[i] for i in sort_indices
]

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.25  # Adjust width for three bars

fig, axs = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [4, 1]}
)

# Plot MCMC data (grouped bars for each label, sorted)
rects1 = axs[0].bar(
    x - width,
    mcmc_base_means_sorted,
    width,
    yerr=mcmc_base_std_devs_sorted,
    capsize=5,
    label="No Ablation",
)
rects2 = axs[0].bar(
    x,
    mcmc_pca_interpreted_means_sorted,
    width,
    yerr=mcmc_pca_interpreted_std_devs_sorted,
    capsize=5,
    label="Interpreted PCA",
)
rects3 = axs[0].bar(
    x + width,
    mcmc_pca_top_intervention_means_sorted,
    width,
    yerr=mcmc_pca_top_intervention_std_devs_sorted,
    capsize=5,
    label="Top 5 PCA",
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
axs[0].tick_params(axis="y", labelsize=14)

# --- Gender Plotting ---
gender_labels = ["Gender"]
x_gender = np.arange(len(gender_labels))

gender_width = 0.15

rects4 = axs[1].bar(
    x_gender - gender_width,
    [gender_base_means],
    gender_width,
    yerr=[gender_base_std_devs],
    capsize=5,
)
rects5 = axs[1].bar(
    x_gender,
    [gender_pca_means],
    gender_width,
    yerr=[gender_pca_std_devs],
    capsize=5,
)
rects6 = axs[1].bar(
    x_gender + gender_width,
    [gender_top_pca_means],
    gender_width,
    yerr=[gender_top_pca_std_devs],
    capsize=5,
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xlim(-0.5, 0.5)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].tick_params(axis="y", labelsize=14)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fontsize=14)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.95)
plt.show()
