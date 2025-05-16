# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt

gender_sae = pd.read_csv("gender_sae.csv")
gender_base = pd.read_csv("gender_base.csv")
gender_sae_top = pd.read_csv("gender_sae_top.csv")

mcmc_sae_top = pd.read_csv("mcmc_sae_top.csv")

mcmc_interpreted_sae = pd.read_csv("mcmc_sae.csv")
mcmc_sae = pd.read_csv("mcmc_random_sae.csv")
mcmc_base = pd.read_csv("mcmc_base.csv")

gender_sae_random = gender_sae[gender_sae["random"] == True].copy()
gender_interpreted = gender_sae[gender_sae["random"] == False].copy()
gender_base = gender_base[gender_base["random"] == False]


gender_sae_random_means, gender_sae_random_std_devs = get_gender_data(gender_sae_random)
gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_sae_top_means, gender_sae_top_std_devs = get_gender_data(gender_sae_top)
gender_interpreted_means, gender_interpreted_std_devs = get_gender_data(gender_interpreted)


mcmc_interpreted_sae_means, mcmc_interpreted_sae_std_devs, mcmc_interpreted_sae_labels = get_mcmc_data(mcmc_interpreted_sae)
mcmc_sae_means, mcmc_sae_std_devs, mcmc_sae_labels = get_mcmc_data(mcmc_sae)
mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(mcmc_base)
mcmc_sae_top_means, mcmc_sae_top_std_devs, mcmc_sae_top_labels = get_mcmc_data(mcmc_sae_top)

# %%
import numpy as np

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)
labels_sorted = [mcmc_sae_labels[i] for i in sort_indices]
mcmc_sae_means_sorted = [mcmc_sae_means[i] for i in sort_indices]
mcmc_sae_std_devs_sorted = [mcmc_sae_std_devs[i] for i in sort_indices]
mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]
mcmc_sae_top_means_sorted = [mcmc_sae_top_means[i] for i in sort_indices]
mcmc_sae_top_std_devs_sorted = [mcmc_sae_top_std_devs[i] for i in sort_indices]
mcmc_interpreted_sae_means_sorted = [mcmc_interpreted_sae_means[i] for i in sort_indices]
mcmc_interpreted_sae_std_devs_sorted = [mcmc_interpreted_sae_std_devs[i] for i in sort_indices]

# Sort PCA data using the same indices

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.15  # Adjust width for five bars

fig, axs = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [4, 1]}
)

# Plot MCMC data (grouped bars for each label, sorted)
rects1 = axs[0].bar(
    x - 1.5*width,
    mcmc_base_means_sorted,
    width,
    yerr=mcmc_base_std_devs_sorted,
    capsize=5,
    label="No Intervention",
)
rects2 = axs[0].bar(
    x - 0.5*width,
    mcmc_sae_means_sorted,
    width,
    yerr=mcmc_sae_std_devs_sorted,
    capsize=5,
    label="Random SAEs (Top 10)",
)
rects3 = axs[0].bar(
    x + 0.5*width,
    mcmc_sae_top_means_sorted,
    width,
    yerr=mcmc_sae_top_std_devs_sorted,
    capsize=5,
    label="Top 10 SAEs",
)
rects4 = axs[0].bar(
    x + 1.5*width,
    mcmc_interpreted_sae_means_sorted,
    width,
    yerr=mcmc_interpreted_sae_std_devs_sorted,
    capsize=5,
    label="Interpreted SAEs",
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


rects5 = axs[1].bar(
    x_gender - 1.5*width,
    [gender_base_means],
    width,
    yerr=[gender_base_std_devs],
    capsize=5,
)
rects6 = axs[1].bar(
    x_gender - 0.5*width,
    [gender_sae_random_means],
    width,
    yerr=[gender_sae_random_std_devs],
    capsize=5,
)
rects7 = axs[1].bar(
    x_gender + 0.5*width,
    [gender_sae_top_means],
    width,
    yerr=[gender_sae_top_std_devs],
    capsize=5,
)
rects8 = axs[1].bar(
    x_gender + 1.5*width,
    [gender_interpreted_means],
    width,
    yerr=[gender_interpreted_std_devs],
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
          ncol=4, fontsize=15)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.95)
plt.show()
