import torch as t
from tqdm import tqdm

dataset_a = "sports"
dataset_b = "pronouns"
label = 0

base_path = f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_base_mean.pt"
tuned_path = f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_tuned_mean.pt"

base_mean = t.load(base_path)
tuned_mean = t.load(tuned_path)

mean_diff = tuned_mean - base_mean

latent_dict = {}

for row_idx, layer_idx in tqdm(enumerate(range(0, 32, 4))):
    base_acts = base_mean[row_idx]
    tuned_acts = tuned_mean[row_idx]

    mean_diff = tuned_acts - base_acts

    top_twenty = t.topk(mean_diff, k=20)

    latent_dict[f"model.layers.{layer_idx}"] = top_twenty.indices.tolist()

t.save(latent_dict, "/root/latent_dict.pt")







# t.save(latent_dict, f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_tuned_mean_diff_latent.pt")


# t.save(mean_diff, f"/workspace/llama_mcmc_base_caches/{dataset_a}_{dataset_b}_{label}_tuned_mean_diff.pt")