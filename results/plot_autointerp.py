# %%

import pandas as pd
import matplotlib.pyplot as plt
from plot_utils import get_mcmc_data
from scipy import stats

# Load data
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

# Calculate correlations
pca_corr, pca_pval = stats.pearsonr(mcmc_auto_pca_means, mcmc_interpreted_pca_means)
sae_corr, sae_pval = stats.pearsonr(mcmc_auto_sae_means, mcmc_interpreted_sae_means)

# %%

# Create a figure with proper layout    
fig = plt.figure(figsize=(4, 8))

# Function to create scatter plot with error bars
def make_scatter_with_error_bars(subplot_idx, title, corr, pval):
    ax = plt.subplot(2, 1, subplot_idx)
    
    # Plot scatter points
    scatter = plt.scatter(
        mcmc_auto_pca_means,
        mcmc_interpreted_pca_means,
        c=[plt.cm.viridis(i / len(mcmc_auto_pca_means)) for i in range(len(mcmc_auto_pca_means))],
        s=50,
        zorder=3,
        label=f'r = {corr:.3f}'
    )
    
    # Add error bars
    plt.errorbar(
        mcmc_auto_pca_means,
        mcmc_interpreted_pca_means,
        xerr=mcmc_auto_pca_std_devs,
        yerr=mcmc_interpreted_pca_std_devs,
        fmt='none',
        ecolor='gray',
        elinewidth=0.8,
        capsize=3,
        alpha=0.5,
        zorder=2
    )
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, zorder=1)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Auto Performance (mean)")
    plt.ylabel("Interpreted Performance (mean)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Make the plot square
    ax.set_aspect('equal', adjustable='box')
    
    return ax

# Create first square plot with error bars
make_scatter_with_error_bars(1, "PCA Interpreted vs Auto Interpreted", pca_corr, pca_pval)

# Create second square plot with error bars
ax = plt.subplot(2, 1, 2)
scatter = plt.scatter(
    mcmc_auto_sae_means,
    mcmc_interpreted_sae_means,
    c=[plt.cm.viridis(i / len(mcmc_auto_sae_means)) for i in range(len(mcmc_auto_sae_means))],
    s=50,
    zorder=3,
    label=f'r = {sae_corr:.3f}'
)

# Add error bars
plt.errorbar(
    mcmc_auto_sae_means,
    mcmc_interpreted_sae_means,
    xerr=mcmc_auto_sae_std_devs,
    yerr=mcmc_interpreted_sae_std_devs,
    fmt='none',
    ecolor='gray',
    elinewidth=0.8,
    capsize=3,
    alpha=0.5,
    zorder=2
)

# Add diagonal line
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, zorder=1)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Auto Performance (mean)")
plt.ylabel("Interpreted Performance (mean)")
plt.title("SAE Interpreted vs Auto Interpreted")
plt.grid(True, alpha=0.3)
plt.legend()

# Adjust layout and add space between subplots
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.show()

# %%

# %%
