# %%
import pandas as pd
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt
import numpy as np

gender_base = pd.read_csv("gender_base.csv")
gender_pca = pd.read_csv("gender_pca.csv")

gender_base = gender_base[gender_base["random"] == False].copy()
gender_pca_interpreted = gender_pca[gender_pca["intervention"] == "interpreted"].copy()
gender_pca_test_only = gender_pca[gender_pca["intervention"] == "test_only"].copy()

gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_pca_interpreted_means, gender_pca_interpreted_std_devs = get_gender_data(
    gender_pca_interpreted
)
gender_pca_test_only_means, gender_pca_test_only_std_devs = get_gender_data(
    gender_pca_test_only
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

mcmc_pca_test_only = mcmc_pca[mcmc_pca["intervention"] == "test_only"].copy()
(
    mcmc_pca_test_only_means,
    mcmc_pca_test_only_std_devs,
    mcmc_pca_test_only_labels,
) = get_mcmc_data(mcmc_pca_test_only)

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

mcmc_pca_test_only_means_sorted = [
    mcmc_pca_test_only_means[i] for i in sort_indices
]
mcmc_pca_test_only_std_devs_sorted = [
    mcmc_pca_test_only_std_devs[i] for i in sort_indices
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
    mcmc_pca_test_only_means_sorted,
    width,
    yerr=mcmc_pca_test_only_std_devs_sorted,
    capsize=5,
    label="Test-time only Ablations",
)
axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
# axs[0].legend(fontsize=14)
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
    [gender_pca_test_only_means],
    gender_width,
    yerr=[gender_pca_test_only_std_devs],
    capsize=5,
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xlim(-0.75, 0.75)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
# axs[1].legend(loc="upper left", fontsize=14)
axs[1].tick_params(axis="y", labelsize=14)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fontsize=15)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.95)
plt.show()

