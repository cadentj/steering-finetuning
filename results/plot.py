# %%

import pandas as pd
import matplotlib.pyplot as plt



def get_mcmc_data():
    full_df = pd.read_csv("mcmc_sae.csv")
    full_df.head()

    # rename dataset_a or dataset_b rows with SVA to verbs
    full_df.loc[full_df['dataset_a'] == 'SVA', 'dataset_a'] = 'verbs'
    full_df.loc[full_df['dataset_b'] == 'SVA', 'dataset_b'] = 'verbs'

    # new column which is tuple of dataset_a and dataset_b
    full_df['pair'] = full_df.apply(lambda row: (row['dataset_a'], row['dataset_b']), axis=1)
    grouped_df = full_df.groupby("pair")

    keep = [
        ("verbs", "sentiment", 1),
        ("sports", "pronouns", 0),
        ("pronouns", "sports", 0),
        ("sentiment", "verbs", 1),
        ("sentiment", "sports", 0),
        ("verbs", "sports", 0),
        ("sentiment", "pronouns", 0),
        ("verbs", "pronouns", 0),
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
        intended_df = df[df['ablated_dataset'] == ablated_dataset]

        mean_accuracy = intended_df['test_accuracy'].mean()
        std_accuracy = intended_df['test_accuracy'].std()
        
        means.append(mean_accuracy)
        std_devs.append(std_accuracy)

        labels.append(f"{dataset_pair[0]} | {dataset_pair[1]}")

    return means, std_devs, labels

def get_gender_data():
    full_df = pd.read_csv("gender_sae.csv")
    full_df.head()

    relevant = full_df[full_df['random'] == False]

    means = relevant['test_accuracy'].mean()
    std_devs = relevant['test_accuracy'].std()

    return means, std_devs

gender_means, gender_std_devs = get_gender_data()
means, std_devs, labels = get_mcmc_data()

# Create bar chart
fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]}) # Adjust figure size and subplot ratios

# Plot MCMC data on the first subplot
axs[0].bar(labels, means, yerr=std_devs, capsize=5) # Add error bars
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
axs[0].set_ylabel('Test Accuracy')
axs[0].set_ylim(0, 1)
axs[0].tick_params(axis='x', rotation=25) # Rotate x-axis labels
axs[0].set_title('MCMC SAE Ablation Accuracy')

# Plot Gender data on the second subplot
axs[1].bar(["Gender"], [gender_means], yerr=[gender_std_devs], capsize=5)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)
axs[1].set_ylabel('Test Accuracy')
axs[1].set_ylim(0, 1)
axs[1].set_title('Gender SAE Ablation Accuracy')


plt.tight_layout() # Adjust layout to prevent labels from overlapping
plt.show()

