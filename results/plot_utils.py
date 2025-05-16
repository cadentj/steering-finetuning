import ast
import numpy as np

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
    std_errors = []
    labels = []

    # Loop through pairs and respective grouped df
    grouped_df = full_df.groupby("pair")
    for dataset_a, dataset_b, label in keep:

        # Find the matching pair in the keep list
        matching_df = None
        for dataset_pair, df in grouped_df:
            if dataset_a == dataset_pair[0] and dataset_b == dataset_pair[1]:
                matching_df = df
                break
        
        # If no matching pair, skip
        if matching_df is None:
            continue

        # Get the ablated dataset
        ablated_dataset = dataset_b if label == 0 else dataset_a

        # Just use test_accuracy by default
        which = "test_accuracy"
        mean_accuracy = None
        std_error = None

        # MCMC plots with interventions
        if "ablated_dataset" in df.columns:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]

            # My PCA stuff uses "deployed_accuracy" to indicate testing w/o intervention
            if "deployed_accuracy" in intended_df.columns:
                # NOTE: Double check this
                if label == 0:
                    which = "deployed_accuracy"
                else:
                    which = "deployed_accuracy_flipped"
            
            # Helena's SAE stuff uses this column to indicate testing w/o intervention
            elif label == 1:
                which = "test_accuracy_flipped"
        
        # MCMC plots w/o interventions
        else:
            intended_df = df

            if label == 1:
                which = "test_accuracy_flipped"       


        if intended_df.shape[0] != 5:
            print(intended_df)
        assert intended_df.shape[0] == 5
        
        mean_accuracy = intended_df[which].mean()
        std_error = intended_df[which].std() / np.sqrt(5)

        means.append(mean_accuracy)
        std_errors.append(std_error)

        if label == 1:
            labels.append(f"{dataset_a} | $\\mathbf{{{dataset_b}}}$")
        else:
            labels.append(f"$\\mathbf{{{dataset_a}}}$ | {dataset_b}")

    return means, std_errors, labels


def get_gender_data(full_df):
    if "deployed_accuracy" in full_df.columns:
        means = full_df["deployed_accuracy"].mean()
        std_errors = full_df["deployed_accuracy"].std() / np.sqrt(len(full_df))
    else:
        means = full_df["test_accuracy"].mean()
        std_errors = full_df["test_accuracy"].std() / np.sqrt(len(full_df))

    return means, std_errors

def get_mcmc_data_all(full_df):
    assert "pair" in full_df.columns, "pair column not found"
    full_df["pair"] = full_df["pair"].apply(ast.literal_eval)

    if "wb_run_group" not in full_df.columns:
        # This is an sae dataset
        assert "ablated_dataset" in full_df.columns

        def make_wb_run_group(row):
            dataset_a = row["pair"][0]
            dataset_b = row["pair"][1]
            label = 0 if row["ablated_dataset"] == dataset_b else 1
            return f"{dataset_a}_{dataset_b}_{label}"

        full_df["wb_run_group"] = full_df.apply(make_wb_run_group, axis=1)

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
        
        ("sentiment", "verbs", 0),
        ("pronouns", "sports", 1),
        ("sports", "pronouns", 1),
        ("verbs", "sentiment", 0),
        ("sports", "sentiment", 1),
        ("sports", "verbs", 1),
        ("pronouns", "sentiment", 1),
        ("pronouns", "verbs", 1),
    ]

    means = []
    std_errors = []
    labels = []

    # Loop through pairs and respective grouped df
    grouped_df = full_df.groupby("wb_run_group")
    for dataset_a, dataset_b, label in keep:

        # Find the matching pair in the keep list
        matching_df = None
        for dataset_pair, df in grouped_df:
            parts = dataset_pair.split("_")

            if dataset_a == parts[0] and dataset_b == parts[1] and label == int(parts[2]):
                matching_df = df
                break

        # If no matching pair, skip
        if matching_df is None:
            continue

        # Get the ablated dataset
        ablated_dataset = dataset_b if label == 0 else dataset_a

        # Just use test_accuracy by default
        which = "test_accuracy"
        mean_accuracy = None
        std_error = None

        print(matching_df.shape, dataset_a, dataset_b, label)
        print(ablated_dataset)
        print(df)
        print(df[df["ablated_dataset"] == ablated_dataset])

        # MCMC plots with interventions
        if "ablated_dataset" in df.columns:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]

            # My PCA stuff uses "deployed_accuracy" to indicate testing w/o intervention
            if "deployed_accuracy" in intended_df.columns:
                # NOTE: Double check this
                if label == 0:
                    which = "deployed_accuracy"
                else:
                    which = "deployed_accuracy_flipped"
            
            # Helena's SAE stuff uses this column to indicate testing w/o intervention
            elif label == 1:
                which = "test_accuracy_flipped"
        
        # MCMC plots w/o interventions
        else:
            intended_df = df

            if label == 1:
                which = "test_accuracy_flipped"       

        assert intended_df.shape[0] == 5
        
        mean_accuracy = intended_df[which].mean()
        std_error = intended_df[which].std() / np.sqrt(5)

        means.append(mean_accuracy)
        std_errors.append(std_error)

        if label == 1:
            labels.append(f"{dataset_a} | $\\mathbf{{{dataset_b}}}$")
        else:
            labels.append(f"$\\mathbf{{{dataset_a}}}$ | {dataset_b}")

    return means, std_errors, labels



def get_mcmc_data_all_base(full_df):
    assert "pair" in full_df.columns, "pair column not found"
    full_df["pair"] = full_df["pair"].apply(ast.literal_eval)

    assert "wb_run_group" not in full_df.columns, "check data version"
    assert "ablated_dataset" not in full_df.columns, "check data version"

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

        ("sentiment", "verbs", 0),
        ("pronouns", "sports", 1),
        ("sports", "pronouns", 1),
        ("verbs", "sentiment", 0),
        ("sports", "sentiment", 1),
        ("sports", "verbs", 1),
        ("pronouns", "sentiment", 1),
        ("pronouns", "verbs", 1),
    ]

    means = []
    std_errors = []
    labels = []

    # Loop through pairs and respective grouped df
    grouped_df = full_df.groupby("pair")
    for dataset_a, dataset_b, label in keep:


        # Find the matching pair in the keep list
        matching_df = None
        for dataset_pair, df in grouped_df:
            if dataset_a == dataset_pair[0] and dataset_b == dataset_pair[1]:
                matching_df = df
                break
        
        # If no matching pair, skip
        if matching_df is None:
            continue

        # Get the ablated dataset
        ablated_dataset = dataset_pair[1 if label else 0]

        # Just use test_accuracy by default
        which = "test_accuracy"
        mean_accuracy = None
        std_error = None

        # MCMC plots with interventions
        if "ablated_dataset" in df.columns:
            intended_df = df[df["ablated_dataset"] == ablated_dataset]

            # My PCA stuff uses "deployed_accuracy" to indicate testing w/o intervention
            if "deployed_accuracy" in intended_df.columns:
                # NOTE: Double check this
                if label == 0:
                    which = "deployed_accuracy"
                else:
                    which = "deployed_accuracy_flipped"
            
            # Helena's SAE stuff uses this column to indicate testing w/o intervention
            elif label == 1:
                which = "test_accuracy_flipped"
        
        # MCMC plots w/o interventions
        else:
            intended_df = df

            if label == 1:
                which = "test_accuracy_flipped"       

        assert intended_df.shape[0] == 5
        
        mean_accuracy = intended_df[which].mean()
        std_error = intended_df[which].std() / np.sqrt(5)

        means.append(mean_accuracy)
        std_errors.append(std_error)

        if label == 1:
            labels.append(f"{dataset_a} | $\\mathbf{{{dataset_b}}}$")
        else:
            labels.append(f"$\\mathbf{{{dataset_a}}}$ | {dataset_b}")

    return means, std_errors, labels

