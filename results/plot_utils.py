def get_mcmc_data(full_df):
    assert "pair" not in full_df.columns

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
            if "no_interventions_accuracy" in intended_df.columns:
                if matching[2] == 0:
                    which = "no_interventions_accuracy"
                else:
                    which = "no_interventions_accuracy_flipped"
            
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


def get_gender_data(full_df):
    if "no_interventions_accuracy" in full_df.columns:
        means = full_df["no_interventions_accuracy"].mean()
        std_devs = full_df["no_interventions_accuracy"].std()
    else:
        means = full_df["test_accuracy"].mean()
        std_devs = full_df["test_accuracy"].std()

    return means, std_devs
