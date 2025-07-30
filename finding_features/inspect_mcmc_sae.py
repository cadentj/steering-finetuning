# %%


from autointerp.vis.dashboard import make_feature_display
import torch as t

task = "pronouns_verbs_1"

# todo
# verbs_sentiment_0

features = f"{task}"
latent_filter = t.load(
    f"/workspace/llama_1b_saes/{task}.pt"
)

latent_filter = {
    name : latents[:20] for name, latents in latent_filter.items()
}

cache_dirs = [
    f"/workspace/llama_1b_sae_caches/{features}_cache/{hookpoint}"
    for hookpoint in latent_filter.keys()
]

cache_dirs


feature_display = make_feature_display(
    cache_dirs,
    latent_filter,
    max_examples=10,
    ctx_len=16,
    load_min_activating=False,
)

