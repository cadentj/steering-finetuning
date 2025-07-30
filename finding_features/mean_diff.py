import torch as t
from tqdm import tqdm

dataset_a = "sports"
dataset_b = "pronouns"
label = 0


for version in ["base", "tuned"]:
    if version == "base":
        path = f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_cache"
    else:
        path = f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_tuned_cache"

    WIDTH = 32768

    layer_idxs = range(0, 32, 4)

    avg_feature_acts = t.zeros(len(list(layer_idxs)), WIDTH)

    for row_idx, layer_idx in tqdm(enumerate(layer_idxs)):
        for shard in range(2):
            shard_path = f"{path}/model.layers.{layer_idx}/{shard}.pt"
            shard = t.load(shard_path)

            activations = shard["activations"]
            locations = shard["locations"]

            print(locations.shape)

            feature_indices = locations[:, 2]  # shape (n,)

            # Sum activations for each feature, ensuring all 32768 features are represented
            activation_sums = t.bincount(
                feature_indices, weights=activations, minlength=32768
            )

            # Count occurrences of each feature
            feature_counts = t.bincount(feature_indices, minlength=32768)

            # Compute average activation per feature
            # Features that didn't appear will have count=0, so we set their average to 0
            average_activations = t.where(
                feature_counts > 0,
                activation_sums / feature_counts.float(),
                t.zeros_like(activation_sums),
            )

            avg_feature_acts[row_idx] += average_activations

    avg_feature_acts = avg_feature_acts / 2

    t.save(avg_feature_acts, f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_{version}_mean.pt")
