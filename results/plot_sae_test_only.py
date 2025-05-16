# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt

gender_sae = pd.read_csv("gender_sae.csv")
gender_sae_test_only = pd.read_csv("gender_sae_autointerp.csv")
gender_base = pd.read_csv("gender_base.csv")

mcmc_interpreted_sae = pd.read_csv("mcmc_sae.csv")
mcmc_base = pd.read_csv("mcmc_base.csv")
mcmc_sae_test_only = pd.read_csv("mcmc_sae_new.csv")

gender_interpreted = gender_sae[gender_sae["random"] == False].copy()
gender_base = gender_base[gender_base["random"] == False]

gender_sae_test_only = gender_sae_test_only[gender_sae_test_only["intervention"] == "test_only"].copy()

gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_interpreted_means, gender_interpreted_std_devs = get_gender_data(gender_interpreted)
gender_test_only_means, gender_test_only_std_devs = get_gender_data(gender_sae_test_only)

mcmc_interpreted_sae_means, mcmc_interpreted_sae_std_devs, mcmc_interpreted_sae_labels = get_mcmc_data(mcmc_interpreted_sae)
mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(mcmc_base)

mcmc_test_only = mcmc_sae_test_only[mcmc_sae_test_only["intervention"] == "test_only"].copy()
mcmc_sae_test_only_means, mcmc_sae_test_only_std_devs, mcmc_sae_test_only_labels = get_mcmc_data(mcmc_test_only)

# %%

for mbaselabel, msaelabel, mtestlabel in zip(mcmc_base_labels, mcmc_interpreted_sae_labels, mcmc_sae_test_only_labels):
    assert mbaselabel == msaelabel == mtestlabel

# %%
import numpy as np

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)
labels_sorted = [mcmc_base_labels[i] for i in sort_indices]
mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]
mcmc_interpreted_sae_means_sorted = [mcmc_interpreted_sae_means[i] for i in sort_indices]
mcmc_interpreted_sae_std_devs_sorted = [mcmc_interpreted_sae_std_devs[i] for i in sort_indices]
mcmc_test_only_means_sorted = [mcmc_sae_test_only_means[i] for i in sort_indices]
mcmc_test_only_std_devs_sorted = [mcmc_sae_test_only_std_devs[i] for i in sort_indices]

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
    mcmc_interpreted_sae_means_sorted,
    width,
    yerr=mcmc_interpreted_sae_std_devs_sorted,
    capsize=5,
    label="Fine-tuning with Ablations",
)
rects3 = axs[0].bar(
    x + width,
    mcmc_test_only_means_sorted,
    width,
    yerr=mcmc_test_only_std_devs_sorted,
    capsize=5,
    label="Test-time only Ablations",
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=14)
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
)
rects5 = axs[1].bar(
    x_gender,
    [gender_interpreted_means],
    width,
    yerr=[gender_interpreted_std_devs],
    capsize=5,
)
rects6 = axs[1].bar(
    x_gender + width,
    [gender_test_only_means],
    width,
    yerr=[gender_test_only_std_devs],
    capsize=5,
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xlim(-0.75, 0.75)
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
