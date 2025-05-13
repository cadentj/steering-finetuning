# %%

def get_mcmc_data_test_time(full_df):
    # assert "pair" not in full_df.columns

    # new column which is tuple of dataset_a and dataset_b
    full_df["pair"] = full_df.apply(
        lambda row: (row["dataset_a"], row["dataset_b"]), axis=1
    )
    grouped_df = full_df.groupby("pair")

    keep = [
        ("SVA", "sentiment", 1),
        ("sports", "pronouns", 0),
        ("pronouns", "sports", 0),
        ("sentiment", "SVA", 1),
        ("sentiment", "sports", 0),
        ("SVA", "sports", 0),
        ("sentiment", "pronouns", 0),
        ("SVA", "pronouns", 0),
    ]

    means = []
    std_devs = []
    labels = []

    for dataset_pair, df in grouped_df:
        matching = None
        for pair in keep:
            if pair[0] == dataset_pair[0] and pair[1] == dataset_pair[1]:
                matching = pair
                break

        if matching is None:
            continue

        ablated_dataset = dataset_pair[1 if matching[2] == 0 else 0]

        which = "test_accuracy"
        mean_accuracy = None
        std_accuracy = None

        try:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]

            # My PCA stuff has this column, 
            if "test_accuracy" in intended_df.columns:
                if matching[2] == 0:
                    which = "test_accuracy"
                else:
                    which = "test_accuracy_flipped"
            
            # Helena's SAE stuff uses this column
            elif matching[2] == 1:
                which = "test_accuracy_flipped"
            
        except Exception as e:
            intended_df = df

            if matching[2] == 1:
                which = "test_accuracy_flipped"       

        mean_accuracy = intended_df[which].mean()
        std_accuracy = intended_df[which].std()

        means.append(mean_accuracy)
        std_devs.append(std_accuracy)

        first = "verbs" if dataset_pair[0] == "SVA" else dataset_pair[0]
        second = "verbs" if dataset_pair[1] == "SVA" else dataset_pair[1]

        if matching[2] == 1:
            labels.append(f"{first} | $\\mathbf{{{second}}}$")
        else:
            labels.append(f"$\\mathbf{{{first}}}$ | {second}")

    return means, std_devs, labels


def get_gender_data_test_time(full_df):
    if "test_accuracy" in full_df.columns:
        means = full_df["test_accuracy"].mean()
        std_devs = full_df["test_accuracy"].std()
    else:
        means = full_df["test_accuracy"].mean()
        std_devs = full_df["test_accuracy"].std()

    return means, std_devs


# %%
import pandas as pd 
from plot_utils import get_gender_data, get_mcmc_data
import matplotlib.pyplot as plt
import numpy as np

gender_pca = pd.read_csv("gender_pca.csv")
gender_base = pd.read_csv("gender_base.csv")

gender_pca = gender_pca[gender_pca["intervention"] == "none"]
gender_base = gender_base[gender_base["random"] == False]



gender_pca_means, gender_pca_std_devs = get_gender_data(gender_pca)
gender_base_means, gender_base_std_devs = get_gender_data(gender_base)
gender_pca_test_time_means, gender_pca_test_time_std_devs = get_gender_data_test_time(gender_pca)


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

mcmc_pca_test_time_means, mcmc_pca_test_time_std_devs, mcmc_pca_test_time_labels = get_mcmc_data_test_time(mcmc_pca_no_intervention)




# %%

# Sort MCMC bars by mean base accuracy (descending)
sort_indices = np.argsort(mcmc_base_means)[::-1]
labels_sorted = [mcmc_pca_no_intervention_labels[i] for i in sort_indices]

mcmc_base_means_sorted = [mcmc_base_means[i] for i in sort_indices]
mcmc_base_std_devs_sorted = [mcmc_base_std_devs[i] for i in sort_indices]

# Sort PCA data using the same indices
mcmc_pca_no_intervention_means_sorted = [mcmc_pca_no_intervention_means[i] for i in sort_indices]
mcmc_pca_no_intervention_std_devs_sorted = [mcmc_pca_no_intervention_std_devs[i] for i in sort_indices]

mcmc_pca_test_time_means_sorted = [mcmc_pca_test_time_means[i] for i in sort_indices]
mcmc_pca_test_time_std_devs_sorted = [mcmc_pca_test_time_std_devs[i] for i in sort_indices]

# Create grouped bar chart for MCMC and Gender data

# --- MCMC Plotting ---
labels = labels_sorted
x = np.arange(len(labels))
width = 0.25  # Adjust width for three bars

fig, axs = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3.5, 1]}
)

# Plot MCMC data (grouped bars for each label, sorted)
rects1 = axs[0].bar(
    x - width,
    mcmc_base_means_sorted,
    width,
    yerr=mcmc_base_std_devs_sorted,
    capsize=5,
    label="Base Performance",
)
rects2 = axs[0].bar(
    x,
    mcmc_pca_no_intervention_means_sorted,
    width,
    yerr=mcmc_pca_no_intervention_std_devs_sorted,
    capsize=5,
    label="No Intervention",
)
rects3 = axs[0].bar(
    x + width,
    mcmc_pca_test_time_means_sorted,
    width,
    yerr=mcmc_pca_test_time_std_devs_sorted,
    capsize=5,
    label="Test Time",
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

gender_width = 0.2

rects5 = axs[1].bar(
    x_gender - gender_width,
    [gender_base_means],
    gender_width,
    yerr=[gender_base_std_devs],
    capsize=5,
    label="Base Performance",
)
rects6 = axs[1].bar(
    x_gender,
    [gender_pca_means],
    gender_width,
    yerr=[gender_pca_std_devs],
    capsize=5,
    label="No Intervention",
)
rects7 = axs[1].bar(
    x_gender + gender_width,
    [gender_pca_test_time_means],
    gender_width,
    yerr=[gender_pca_test_time_std_devs],
    capsize=5,
    label="Test Time",
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