# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt
import numpy as np

gender_pca = pd.read_csv("gender_pca.csv")
gender_base = pd.read_csv("gender_base.csv")

gender_pca = gender_pca[gender_pca["intervention"] == "none"]
gender_base = gender_base[gender_base["random"] == False]

# Fix the filtering for random and top interventions
gender_random_pca = pd.read_csv("gender_pca.csv")
gender_random_pca = gender_random_pca[gender_random_pca["intervention"] == "random_intervention"]

gender_top_pca = pd.read_csv("gender_pca.csv")
gender_top_pca = gender_top_pca[gender_top_pca["intervention"] == "top_intervention"]

gender_pca_means, gender_pca_std_devs = get_gender_data(gender_pca)
gender_base_means, gender_base_std_devs = get_gender_data(gender_base)

gender_random_pca_means, gender_random_pca_std_devs = get_gender_data(gender_random_pca)
gender_top_pca_means, gender_top_pca_std_devs = get_gender_data(gender_top_pca)

mcmc_base = pd.read_csv("mcmc_base.csv")
mcmc_base_means, mcmc_base_std_devs, mcmc_base_labels = get_mcmc_data(mcmc_base)


mcmc_pca = pd.read_csv("mcmc_pca.csv")
mcmc_pca_no_intervention = mcmc_pca[mcmc_pca["intervention"] == "none"]

# drop pair column
mcmc_pca_no_intervention = mcmc_pca_no_intervention.drop(columns=["pair"])

# rename verbs to SVA
mcmc_pca_no_intervention = mcmc_pca_no_intervention.replace("verbs", "SVA")

mcmc_pca_no_intervention_means, mcmc_pca_no_intervention_std_devs, mcmc_pca_no_intervention_labels = get_mcmc_data(
    mcmc_pca_no_intervention
)


# %%

mcmc_pca_random_intervention = mcmc_pca[mcmc_pca["intervention"] == "random_intervention"]

# drop pair column
mcmc_pca_random_intervention = mcmc_pca_random_intervention.drop(columns=["pair"])

# rename verbs to SVA
mcmc_pca_random_intervention = mcmc_pca_random_intervention.replace("verbs", "SVA")

mcmc_pca_random_intervention_means, mcmc_pca_random_intervention_std_devs, mcmc_pca_random_intervention_labels = get_mcmc_data(
    mcmc_pca_random_intervention
)


mcmc_pca_top_intervention = mcmc_pca[mcmc_pca["intervention"] == "top_intervention"]

# drop pair column
mcmc_pca_top_intervention = mcmc_pca_top_intervention.drop(columns=["pair"])

# rename verbs to SVA
mcmc_pca_top_intervention = mcmc_pca_top_intervention.replace("verbs", "SVA")

mcmc_pca_top_intervention_means, mcmc_pca_top_intervention_std_devs, mcmc_pca_top_intervention_labels = get_mcmc_data(
    mcmc_pca_top_intervention
)


# %%

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)[::-1]
labels_sorted = [mcmc_pca_random_intervention_labels[i] for i in sort_indices]

mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]

# Sort PCA data using the same indices
mcmc_pca_no_intervention_means_sorted = [mcmc_pca_no_intervention_means[i] for i in sort_indices]
mcmc_pca_no_intervention_std_devs_sorted = [mcmc_pca_no_intervention_std_devs[i] for i in sort_indices]

mcmc_pca_random_intervention_means_sorted = [mcmc_pca_random_intervention_means[i] for i in sort_indices]
mcmc_pca_random_intervention_std_devs_sorted = [mcmc_pca_random_intervention_std_devs[i] for i in sort_indices]

mcmc_pca_top_intervention_means_sorted = [mcmc_pca_top_intervention_means[i] for i in sort_indices]
mcmc_pca_top_intervention_std_devs_sorted = [mcmc_pca_top_intervention_std_devs[i] for i in sort_indices]

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.2  # Adjust width for four bars

fig, axs = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3.5, 1]}
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
    mcmc_pca_no_intervention_means_sorted,
    width,
    yerr=mcmc_pca_no_intervention_std_devs_sorted,
    capsize=5,
    label="Interpreted PCA",
)
rects3 = axs[0].bar(
    x + 0.5*width,
    mcmc_pca_random_intervention_means_sorted,
    width,
    yerr=mcmc_pca_random_intervention_std_devs_sorted,
    capsize=5,
    label="Random PCA (Top 20)",
)
rects4 = axs[0].bar(
    x + 1.5*width,
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
axs[0].legend(fontsize=14)
axs[0].tick_params(axis='y', labelsize=14)

# --- Gender Plotting ---
gender_labels = ["Gender"]
x_gender = np.arange(len(gender_labels))


gender_width = 0.1

rects5 = axs[1].bar(
    x_gender - 1.5*(gender_width),
    [gender_base_means],
    gender_width,
    yerr=[gender_base_std_devs],
    capsize=5,
    label="No Intervention",
)
rects6 = axs[1].bar(
    x_gender - 0.5*(gender_width),
    [gender_pca_means],
    gender_width,
    yerr=[gender_pca_std_devs],
    capsize=5,
    label="Interpreted PCA",
)
rects7 = axs[1].bar(
    x_gender + 0.5*(gender_width),
    [gender_random_pca_means],
    gender_width,
    yerr=[gender_random_pca_std_devs],
    capsize=5,
    label="Random PCA (Top 20)",
)
rects8 = axs[1].bar(
    x_gender + 1.5*(gender_width),
    [gender_top_pca_means],
    gender_width,
    yerr=[gender_top_pca_std_devs],
    capsize=5,
    label="Top 5 PCA",
)
axs[1].set_ylabel("Test Accuracy", fontsize=16)
axs[1].set_ylim(0, 1)
axs[1].set_xticks(x_gender)
axs[1].set_xlim(-0.5, .5)
axs[1].set_xticklabels(gender_labels, fontsize=14)
axs[1].grid(axis="y", linestyle="--", alpha=0.7)
axs[1].legend(loc='upper left', fontsize=14)
axs[1].tick_params(axis='y', labelsize=14)

plt.tight_layout()
plt.show()