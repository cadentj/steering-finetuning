# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt
import numpy as np

# Load all data
# SAE data
gender_sae = pd.read_csv("gender_sae.csv")
gender_base = pd.read_csv("gender_base.csv")
gender_sae_top = pd.read_csv("gender_sae_top.csv")
mcmc_sae_top = pd.read_csv("mcmc_sae_top.csv")
mcmc_interpreted_sae = pd.read_csv("mcmc_sae.csv")
mcmc_sae = pd.read_csv("mcmc_random_sae.csv")
mcmc_base = pd.read_csv("mcmc_base.csv")

# PCA data
gender_pca = pd.read_csv("gender_pca.csv")
mcmc_pca = pd.read_csv("mcmc_pca.csv")

# Process SAE data
gender_sae_random = gender_sae[gender_sae["random"] == True].copy()
gender_interpreted_sae = gender_sae[gender_sae["random"] == False].copy()
gender_base = gender_base[gender_base["random"] == False]

# Process PCA data
gender_interpreted_pca = gender_pca[gender_pca["intervention"] == "interpreted"]
gender_random_pca = gender_pca[gender_pca["intervention"] == "random"]
gender_top_pca = gender_pca[gender_pca["intervention"] == "top"]

mcmc_pca_interpreted = mcmc_pca[mcmc_pca["intervention"] == "interpreted"].copy()
mcmc_pca_random = mcmc_pca[mcmc_pca["intervention"] == "random"].copy()
mcmc_pca_top = mcmc_pca[mcmc_pca["intervention"] == "top"].copy()

# Get means and std devs
# SAE data
gender_sae_random_means, gender_sae_random_std_devs = get_gender_data(gender_sae_random)
gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_sae_top_means, gender_sae_top_std_devs = get_gender_data(gender_sae_top)
gender_interpreted_sae_means, gender_interpreted_sae_std_devs = get_gender_data(gender_interpreted_sae)

# PCA data
gender_pca_means, gender_pca_std_devs = get_gender_data(gender_interpreted_pca)
gender_random_pca_means, gender_random_pca_std_devs = get_gender_data(gender_random_pca)
gender_top_pca_means, gender_top_pca_std_devs = get_gender_data(gender_top_pca)

# MCMC data
mcmc_interpreted_sae_means, mcmc_interpreted_sae_std_devs, mcmc_sae_labels = get_mcmc_data(mcmc_interpreted_sae)
mcmc_sae_means, mcmc_sae_std_devs, _ = get_mcmc_data(mcmc_sae)
mcmc_base_means, mcmc_base_std_devs, _ = get_mcmc_data(mcmc_base)
mcmc_sae_top_means, mcmc_sae_top_std_devs, _ = get_mcmc_data(mcmc_sae_top)

mcmc_pca_interpreted_means, mcmc_pca_interpreted_std_devs, _ = get_mcmc_data(mcmc_pca_interpreted)
mcmc_pca_random_means, mcmc_pca_random_std_devs, _ = get_mcmc_data(mcmc_pca_random)
mcmc_pca_top_means, mcmc_pca_top_std_devs, _ = get_mcmc_data(mcmc_pca_top)

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)[::-1]
labels_sorted = [mcmc_sae_labels[i] for i in sort_indices]

# Sort all MCMC data
mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]

mcmc_sae_means_sorted = [mcmc_sae_means[i] for i in sort_indices]
mcmc_sae_std_devs_sorted = [mcmc_sae_std_devs[i] for i in sort_indices]

mcmc_sae_top_means_sorted = [mcmc_sae_top_means[i] for i in sort_indices]
mcmc_sae_top_std_devs_sorted = [mcmc_sae_top_std_devs[i] for i in sort_indices]

mcmc_interpreted_sae_means_sorted = [mcmc_interpreted_sae_means[i] for i in sort_indices]
mcmc_interpreted_sae_std_devs_sorted = [mcmc_interpreted_sae_std_devs[i] for i in sort_indices]

mcmc_pca_interpreted_means_sorted = [mcmc_pca_interpreted_means[i] for i in sort_indices]
mcmc_pca_interpreted_std_devs_sorted = [mcmc_pca_interpreted_std_devs[i] for i in sort_indices]

mcmc_pca_random_means_sorted = [mcmc_pca_random_means[i] for i in sort_indices]
mcmc_pca_random_std_devs_sorted = [mcmc_pca_random_std_devs[i] for i in sort_indices]

mcmc_pca_top_means_sorted = [mcmc_pca_top_means[i] for i in sort_indices]
mcmc_pca_top_std_devs_sorted = [mcmc_pca_top_std_devs[i] for i in sort_indices]

# Define color scheme
# Blues for SAE
sae_interpreted_color = '#b3e0ff'  # Light blue
sae_random_color = '#0088cc'  # Medium blue
sae_top_color = '#005580'  # Dark blue

# Oranges for PCA
pca_interpreted_color = '#ffdcbd'  # Light orange
pca_random_color = '#ff9e5e'  # Medium orange
pca_top_color = '#d86e3a'  # Dark orange

# Gray for base (no ablation)
base_color = '#C0C0C0'  # Medium gray

# Create the plot
fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [4, 1]})

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.12  # Adjust width for multiple bars

# Plot interpreted performance as background bars (skip No Ablation)
# SAE interpreted backgrounds
axs[0].bar(x - width, mcmc_interpreted_sae_means_sorted, width, 
           color=sae_interpreted_color, alpha=0.7, label='Interpreted SAE')
axs[0].bar(x, mcmc_interpreted_sae_means_sorted, width, 
           color=sae_interpreted_color, alpha=0.7)

# PCA interpreted backgrounds
axs[0].bar(x + width, mcmc_pca_interpreted_means_sorted, width, 
           color=pca_interpreted_color, alpha=0.7, label='Interpreted PCA')
axs[0].bar(x + 2*width, mcmc_pca_interpreted_means_sorted, width, 
           color=pca_interpreted_color, alpha=0.7)

# Plot baseline performance
rects1 = axs[0].bar(x - 2*width, mcmc_base_means_sorted, width,
                    yerr=mcmc_base_std_devs_sorted, capsize=5,
                    color=base_color,
                    label='No Ablation')
rects2 = axs[0].bar(x - width, mcmc_sae_means_sorted, width,
                    yerr=mcmc_sae_std_devs_sorted, capsize=5,
                    color=sae_random_color,
                    label='Random SAE')
rects3 = axs[0].bar(x, mcmc_sae_top_means_sorted, width,
                    yerr=mcmc_sae_top_std_devs_sorted, capsize=5,
                    color=sae_top_color,
                    label='Top SAE')
rects4 = axs[0].bar(x + width, mcmc_pca_random_means_sorted, width,
                    yerr=mcmc_pca_random_std_devs_sorted, capsize=5,
                    color=pca_random_color,
                    label='Random PCA')
rects5 = axs[0].bar(x + 2*width, mcmc_pca_top_means_sorted, width,
                    yerr=mcmc_pca_top_std_devs_sorted, capsize=5,
                    color=pca_top_color,
                    label='Top PCA')

axs[0].set_ylabel("Test Accuracy", fontsize=16)
axs[0].set_ylim(0, 1)
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=14)
axs[0].grid(axis="y", linestyle="--", alpha=0.7)
axs[0].tick_params(axis='y', labelsize=14)

# --- Gender Plotting ---
gender_labels = ["Gender"]
x_gender = np.arange(len(gender_labels))

# Plot interpreted performance as background bars (skip No Ablation)
# SAE interpreted backgrounds
axs[1].bar(x_gender - width, [gender_interpreted_sae_means], width,
           color=sae_interpreted_color, alpha=0.7)
axs[1].bar(x_gender, [gender_interpreted_sae_means], width,
           color=sae_interpreted_color, alpha=0.7)

# PCA interpreted backgrounds
axs[1].bar(x_gender + width, [gender_pca_means], width,
           color=pca_interpreted_color, alpha=0.7)
axs[1].bar(x_gender + 2*width, [gender_pca_means], width,
           color=pca_interpreted_color, alpha=0.7)

# Plot baseline performance
axs[1].bar(x_gender - 2*width, [gender_base_means], width,
           yerr=[gender_base_std_devs], capsize=5,
           color=base_color)
axs[1].bar(x_gender - width, [gender_sae_random_means], width,
           yerr=[gender_sae_random_std_devs], capsize=5,
           color=sae_random_color)
axs[1].bar(x_gender, [gender_sae_top_means], width,
           yerr=[gender_sae_top_std_devs], capsize=5,
           color=sae_top_color)
axs[1].bar(x_gender + width, [gender_random_pca_means], width,
           yerr=[gender_random_pca_std_devs], capsize=5,
           color=pca_random_color)
axs[1].bar(x_gender + 2*width, [gender_top_pca_means], width,
           yerr=[gender_top_pca_std_devs], capsize=5,
           color=pca_top_color)

axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xlim(-1, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].tick_params(axis='y', labelsize=14)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=7, fontsize=14)

plt.tight_layout()
# Adjust layout to make room for the legend at the top
plt.subplots_adjust(top=0.9)
plt.show()

