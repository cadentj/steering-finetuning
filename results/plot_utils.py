import ast

def get_mcmc_data(full_df):
    assert "pair" in full_df.columns, "pair column not found"
    full_df["pair"] = full_df["pair"].apply(ast.literal_eval)

    # keep[2] is the index that is intended
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

    # Loop through pairs and respective grouped df
    grouped_df = full_df.groupby("pair")
    for dataset_pair, df in grouped_df:

        # Find the matching pair in the keep list
        matching = None
        for pair in keep:
            if pair[0] == dataset_pair[0] and pair[1] == dataset_pair[1]:
                matching = pair
                break
        
        # If no matching pair, skip
        if matching is None:
            continue

        # Get the ablated dataset
        ablated_dataset = dataset_pair[1 if matching[2] == 0 else 0]

        # Just use test_accuracy by default
        which = "test_accuracy"
        mean_accuracy = None
        std_accuracy = None

        # MCMC plots with interventions
        if "ablated_dataset" in df.columns:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]

            # My PCA stuff uses "deployed_accuracy" to indicate testing w/o intervention
            if "deployed_accuracy" in intended_df.columns:
                # NOTE: Double check this
                if matching[2] == 0:
                    which = "deployed_accuracy"
                else:
                    which = "deployed_accuracy_flipped"
            
            # Helena's SAE stuff uses this column to indicate testing w/o intervention
            elif matching[2] == 1:
                which = "test_accuracy_flipped"
        
        # MCMC plots w/o interventions
        else:
            intended_df = df

            if matching[2] == 1:
                which = "test_accuracy_flipped"       

        print(dataset_pair, intended_df.shape)

        assert intended_df.shape[0] == 5
        
        mean_accuracy = intended_df[which].mean()
        std_accuracy = intended_df[which].std()

        means.append(mean_accuracy)
        std_devs.append(std_accuracy)

        if matching[2] == 1:
            labels.append(f"{dataset_pair[0]} | $\\mathbf{{{dataset_pair[1]}}}$")
        else:
            labels.append(f"$\\mathbf{{{dataset_pair[0]}}}$ | {dataset_pair[1]}")

    return means, std_devs, labels


def get_gender_data(full_df):
    if "deployed_accuracy" in full_df.columns:
        means = full_df["deployed_accuracy"].mean()
        std_devs = full_df["deployed_accuracy"].std()
    else:
        means = full_df["test_accuracy"].mean()
        std_devs = full_df["test_accuracy"].std()

    return means, std_devs
