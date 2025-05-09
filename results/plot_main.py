# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt

gender_sae = pd.read_csv("gender_sae.csv")
gender_base = pd.read_csv("gender_base.csv")

mcmc_sae = pd.read_csv("mcmc_sae.csv")
mcmc_base = pd.read_csv("mcmc_base.csv")

gender_sae_means, gender_sae_std_devs = get_gender_data(gender_sae)
gender_base_means, gender_base_std_devs = get_gender_data(gender_base)

# %%
mcmc_sae_means, mcmc_sae_std_devs, mcmc_sae_labels = get_mcmc_data(mcmc_sae)
mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(mcmc_base)
#%%


mcmc_pca = pd.read_csv("mcmc_pca.csv")
mcmc_pca_no_intervention = mcmc_pca[mcmc_pca["intervention"] == "none"]

# drop pair column
mcmc_pca_no_intervention = mcmc_pca_no_intervention.drop(columns=["pair"])

# rename verbs to SVA
mcmc_pca_no_intervention = mcmc_pca_no_intervention.replace("verbs", "SVA")

mcmc_pca_means, mcmc_pca_std_devs, mcmc_pca_labels = get_mcmc_data(
    mcmc_pca_no_intervention
)


# %%
import numpy as np

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)[::-1]
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
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]}
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
    label="CAFT with SAE",
)
rects3 = axs[0].bar(
    x + width,
    mcmc_pca_means_sorted,
    width,
    yerr=mcmc_pca_std_devs_sorted,
    capsize=5,
    label="CAFT with PCA",
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
axs[0].legend(fontsize=14)
axs[0].tick_params(axis='y', labelsize=14)

# --- Gender Plotting ---
gender_labels = ["Gender"]
x_gender = np.arange(len(gender_labels))


rects4 = axs[1].bar(
    x_gender - width / 2,
    [gender_base_means],
    width,
    yerr=[gender_base_std_devs],
    capsize=5,
    label="No Intervention",
)
rects5 = axs[1].bar(
    x_gender + width / 4,
    [gender_sae_means],
    width,
    yerr=[gender_sae_std_devs],
    capsize=5,
    label="CAFT with SAE",
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].legend(loc='lower left', fontsize=14)
axs[1].tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.show()