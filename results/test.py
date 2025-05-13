# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt


# gender_sae = pd.read_csv("gender_sae.csv")
# gender_base = pd.read_csv("gender_base.csv")

mcmc_auto_sae = pd.read_csv("mcmc_auto_sae.csv")
mcmc_sae = pd.read_csv("mcmc_sae.csv")


# gender_sae = gender_sae[gender_sae["random"] == False]
# gender_base = gender_base[gender_base["random"] == False]

# gender_sae_means, gender_sae_std_devs = get_gender_data(gender_sae)
# gender_base_means, gender_base_std_devs = get_gender_data(gender_base)


mcmc_auto_sae_means, mcmc_auto_sae_std_devs, mcmc_auto_sae_labels = get_mcmc_data(mcmc_auto_sae)
mcmc_sae_means, mcmc_sae_std_devs, mcmc_sae_labels = get_mcmc_data(mcmc_sae)

# %%

# Scatter plot: x = auto, y = sae
plt.figure(figsize=(5,5))

# Define a list of marker styles to cycle through
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
num_points = len(mcmc_auto_sae_means)

for i in range(num_points):
    plt.scatter(mcmc_auto_sae_means[i], mcmc_sae_means[i],
                c=[plt.cm.viridis(i / num_points)],
                s=100, marker=markers[i % len(markers)], label=None)
    plt.annotate(mcmc_auto_sae_labels[i], (mcmc_auto_sae_means[i], mcmc_sae_means[i]), fontsize=8, ha='right')

plt.xlabel('Auto Performance (mean)')
plt.ylabel('SAE Performance (mean)')
plt.title('Auto vs SAE Performance')
plt.grid(True)
plt.tight_layout()
plt.show()
