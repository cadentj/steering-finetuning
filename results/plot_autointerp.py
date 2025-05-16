# %%

import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import get_mcmc_data, get_gender_data
from scipy import stats

# Load data
gender_pca = pd.read_csv("gender_pca_autointerp.csv")
gender_pca_autointerp = gender_pca[gender_pca["intervention"] == "autointerp"].copy()
gender_pca_interpreted = gender_pca[gender_pca["intervention"] == "interpreted"].copy()

gender_sae = pd.read_csv("gender_sae_autointerp.csv")
gender_sae_interpreted = pd.read_csv("gender_sae.csv")

gender_sae_interpreted = gender_sae_interpreted[gender_sae_interpreted["random"] == False].copy()
gender_sae_autointerp = gender_sae[gender_sae["intervention"] == "autointerp"].copy()


gender_sae_means, gender_sae_std_devs = get_gender_data(gender_sae_interpreted)
gender_pca_means, gender_pca_std_devs = get_gender_data(gender_pca_interpreted)
gender_pca_autointerp_means, gender_pca_autointerp_std_devs = get_gender_data(gender_pca_autointerp)
gender_sae_autointerp_means, gender_sae_autointerp_std_devs = get_gender_data(gender_sae_autointerp)



# %%

mcmc_pca = pd.read_csv("mcmc_pca.csv")
mcmc_interpreted_sae = pd.read_csv("mcmc_sae.csv")
mcmc_auto_sae = pd.read_csv("mcmc_autointerp_sae.csv")

# Process data
mcmc_auto_pca = mcmc_pca[mcmc_pca["intervention"] == "autointerp"].copy()
mcmc_interpreted_pca = mcmc_pca[mcmc_pca["intervention"] == "interpreted"].copy()
mcmc_auto_sae = mcmc_auto_sae[mcmc_auto_sae["intervention"] == "autointerp"].copy()


# Get values
mcmc_auto_pca_means, mcmc_auto_pca_std_devs, mcmc_auto_pca_labels = get_mcmc_data(mcmc_auto_pca)
mcmc_interpreted_pca_means, mcmc_interpreted_pca_std_devs, mcmc_interpreted_pca_labels = get_mcmc_data(mcmc_interpreted_pca)

mcmc_auto_sae_means, mcmc_auto_sae_std_devs, mcmc_auto_sae_labels = get_mcmc_data(mcmc_auto_sae)
mcmc_interpreted_sae_means, mcmc_interpreted_sae_std_devs, mcmc_interpreted_sae_labels = get_mcmc_data(mcmc_interpreted_sae)



# %%
import numpy as np
all_auto_pca_means = np.append(mcmc_auto_pca_means, gender_pca_autointerp_means)
all_auto_pca_std_devs = np.append(mcmc_auto_pca_std_devs, gender_pca_autointerp_std_devs)

all_auto_sae_means = np.append(mcmc_auto_sae_means, gender_sae_autointerp_means)
all_auto_sae_std_devs = np.append(mcmc_auto_sae_std_devs, gender_sae_autointerp_std_devs)

all_interpreted_pca_means = np.append(mcmc_interpreted_pca_means, gender_pca_means)
all_interpreted_pca_std_devs = np.append(mcmc_interpreted_pca_std_devs, gender_pca_std_devs)

all_interpreted_sae_means = np.append(mcmc_interpreted_sae_means, gender_sae_means)
all_interpreted_sae_std_devs = np.append(mcmc_interpreted_sae_std_devs, gender_sae_std_devs)



# Calculate correlations
pca_corr, pca_pval = stats.pearsonr(all_auto_pca_means, all_interpreted_pca_means)
sae_corr, sae_pval = stats.pearsonr(all_auto_sae_means, all_interpreted_sae_means)



# %%

# Create a single figure
plt.figure(figsize=(5, 5))

# Plot PCA data
plt.scatter(
    all_auto_pca_means,
    all_interpreted_pca_means,
    c='blue',
    s=50,
    zorder=3,
    label=f'PCA (r = {pca_corr:.3f})'
)

# Add error bars for PCA
plt.errorbar(
    all_auto_pca_means,
    all_interpreted_pca_means,
    xerr=all_auto_pca_std_devs,
    yerr=all_interpreted_pca_std_devs,
    fmt='none',
    ecolor='blue',
    elinewidth=0.8,
    capsize=3,
    alpha=0.3,
    zorder=2
)

# Plot SAE data
plt.scatter(
    all_auto_sae_means,
    all_interpreted_sae_means,
    c='red',
    s=50,
    zorder=3,
    label=f'SAE (r = {sae_corr:.3f})'
)

# Add error bars for SAE
plt.errorbar(
    all_auto_sae_means,
    all_interpreted_sae_means,
    xerr=all_auto_sae_std_devs,
    yerr=all_interpreted_sae_std_devs,
    fmt='none',
    ecolor='red',
    elinewidth=0.8,
    capsize=3,
    alpha=0.3,
    zorder=2
)

# Add diagonal line
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, zorder=1)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Auto Performance (mean)", fontsize=15)
plt.ylabel("Interpreted Performance (mean)", fontsize=15)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Make the plot square
plt.gca().set_aspect('equal', adjustable='box')

# Increase tick label sizes
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()
plt.show()

# %%

# %%
