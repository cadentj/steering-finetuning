# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data_all, get_mcmc_data_all_base
import matplotlib.pyplot as plt

# Define custom colors
COLOR_BASE = '#86b6d7'  # Light blue
COLOR_SAE = '#C0C0C0'   # Light gray
COLOR_PCA = '#D98A70'   # Light pink

gender_base = pd.read_csv("gender_base.csv")
gender_sae = pd.read_csv("gender_sae.csv")
gender_pca = pd.read_csv("gender_pca.csv")

gender_base = gender_base[gender_base["random"] == False]
gender_sae = gender_sae[gender_sae["random"] == False]
gender_pca = gender_pca[gender_pca["intervention"] == "interpreted"]

gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_sae_means, gender_sae_std_devs = get_gender_data(gender_sae)
gender_pca_means, gender_pca_std_devs = get_gender_data(gender_pca)

mcmc_base = pd.read_csv("mcmc_base.csv")
mcmc_sae = pd.read_csv("mcmc_sae.csv")
mcmc_pca = pd.read_csv("mcmc_pca_all.csv")

# %%

mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data_all_base(mcmc_base)
mcmc_sae_means, mcmc_sae_std_devs, mcmc_sae_labels = get_mcmc_data_all(mcmc_sae)

mcmc_pca = mcmc_pca[mcmc_pca["intervention"] == "interpreted"]
mcmc_pca_means, mcmc_pca_std_devs, mcmc_pca_labels = get_mcmc_data_all(mcmc_pca)

# %%

for pca_label, base_label, sae_label in zip(mcmc_pca_labels, mcmc_base_labels, mcmc_sae_labels):
    assert pca_label == base_label == sae_label

# %%

for base_mean, base_label in zip(mcmc_base_means, mcmc_base_labels):
    if "sports" in base_label and "pronouns" in base_label:
        print(base_label, base_mean)
print("--------------------------------")
for mean, label in zip(mcmc_pca_means, mcmc_pca_labels):
    if "sports" in label and "pronouns" in label:
        print(label, mean)

# %%

sae_beats_base = 0
pca_beats_base = 0
for pca, sae, base in zip(mcmc_pca_means, mcmc_sae_means, mcmc_base_means):
    if pca > base and pca > 0.5:
        pca_beats_base += 1

    if sae > base and sae > 0.5:
        sae_beats_base += 1

print(f"PCA beats base: {pca_beats_base}")
print(f"SAE beats base: {sae_beats_base}")




# %%

import numpy as np

# Sort MCMC bars by mean base accuracy (ascending)
sort_indices = np.argsort(mcmc_base_means)
labels_sorted = [mcmc_sae_labels[i] for i in sort_indices]
mcmc_sae_means_sorted = [mcmc_sae_means[i] for i in sort_indices]
mcmc_sae_std_devs_sorted = [mcmc_sae_std_devs[i] for i in sort_indices]
mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]

# Sort PCA data using the same indices
mcmc_pca_means_sorted = [mcmc_pca_means[i] for i in sort_indices]
mcmc_pca_std_devs_sorted = [mcmc_pca_std_devs[i] for i in sort_indices]

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.25  # Adjust width for three bars

fig, axs = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [8, 1]}
)

# Plot MCMC data (grouped bars for each label, sorted)
rects1 = axs[0].bar(
    x - width,
    mcmc_base_means_sorted,
    width,
    yerr=mcmc_base_std_devs_sorted,
    capsize=5,
    label="No Intervention",
    color=COLOR_BASE
)
rects2 = axs[0].bar(
    x,
    mcmc_sae_means_sorted,
    width,
    yerr=mcmc_sae_std_devs_sorted,
    capsize=5,
    label="CAFT with SAE",
    color=COLOR_SAE
)
rects3 = axs[0].bar(
    x + width,
    mcmc_pca_means_sorted,
    width,
    yerr=mcmc_pca_std_devs_sorted,
    capsize=5,
    label="CAFT with PCA",
    color=COLOR_PCA
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, ha="right", rotation=25, fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
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
    color=COLOR_BASE
)
rects5 = axs[1].bar(
    x_gender,
    [gender_sae_means],
    width,
    yerr=[gender_sae_std_devs],
    capsize=5,
    color=COLOR_SAE
)
rects6 = axs[1].bar(
    x_gender + width,
    [gender_pca_means],
    width,
    yerr=[gender_pca_std_devs],
    capsize=5,
    color=COLOR_PCA
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xlim(-1, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].tick_params(axis='y', labelsize=14)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3)
plt.setp(legend.get_texts(), fontsize=15)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.95)
plt.show()
