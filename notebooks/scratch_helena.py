# %%

import sys

sys.path.append("/root/steering-finetuning")

import torch as t


# %%

path = "/workspace/llama_saes/verbs_sentiment_1.pt"

# %%

feats = t.load(path)

# %%


cache_path = "/workspace/llama_mc/llama_mcmc_sae_caches/verbs_sentiment_1_cache"
module_name = list(feats.keys())[0]
cache_0 = t.load(f"{cache_path}/{module_name}/0.pt")
cache_1 = t.load(f"{cache_path}/{module_name}/1.pt")

# %%
print(feats[module_name])
print(cache_0["locations"][:, 2].unique())
print(cache_1["locations"][:, 2].unique())

# %%

# %%

from autointerp.vis.dashboard import make_feature_display
import torch as t

features = "sentiment_sports_0"
latent_filter = t.load(
    f"/workspace/llama_mc/llama_mcmc_saes/{features}.pt", weights_only=False
)

cache_dirs = [
    f"/workspace/llama_mc/llama_mcmc_sae_caches/{features}_cache/{hookpoint}"
    for hookpoint in latent_filter.keys()
]

# %%

feature_display = make_feature_display(
    cache_dirs,
    latent_filter,
    max_examples=10,
    ctx_len=16,
    load_min_activating=False,
)
# %%
interpreted ={
    '10': [31700],
    '2': [9873]
}