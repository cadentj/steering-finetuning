# %%

from autointerp.vis.dashboard import make_feature_display

layers = list(range(0,26,2))
first_half = layers[:len(layers)//2]
second_half = layers[len(layers)//2:]
hookpoints = [
    f"model.layers.{i}" for i in first_half
]

cache_dirs = [
    f"/root/pca_caches/pronouns_sports_cache/{hookpoint}" for hookpoint in hookpoints
]

features = {
    hookpoint: list(range(20)) for hookpoint in hookpoints
}

feature_display = make_feature_display(cache_dirs, features, max_examples=10, ctx_len=16, load_min_activating=True)

# %%

from collections import defaultdict
import torch as t
from features.mcmc_features import exported_features

def make_intervention(features, pcs):
    intervention = defaultdict(list)
    for hookpoint, pc_indices in features.items():
        for pc_index in pc_indices:
            intervention[hookpoint].append(pcs[hookpoint][:,pc_index])

    intervention = {
        hookpoint: t.stack(pcs).T for hookpoint, pcs in intervention.items()
    }

    return intervention

for name, features in exported_features.items():
    pair = name.split("_features")[0]
    pcs = t.load(f"/root/pcas/{pair}_all.pt")
    intervention = make_intervention(features, pcs)
    t.save(intervention, f"/root/pcas/{name}_intervention.pt")


# %%

from collections import defaultdict
import torch as t
from features.mcmc_features import exported_features
import random

seed = 4
random.seed(seed)

def make_intervention(features, pcs):
    intervention = defaultdict(list)
    for hookpoint, pc_indices in features.items():
        n_pcs = len(pc_indices)
        if n_pcs == 0:
            continue
        random_idxs = random.sample(list(range(20)), n_pcs)
        for idx in random_idxs:
            intervention[hookpoint].append(pcs[hookpoint][:,idx])

    intervention = {
        hookpoint: t.stack(pcs).T for hookpoint, pcs in intervention.items()
    }

    return intervention

for name, features in exported_features.items():
    pair = name.split("_features")[0]
    pcs = t.load(f"/root/pcas/{pair}_all.pt")
    intervention = make_intervention(features, pcs)
    t.save(intervention, f"/root/pcas/{name}_random_intervention_s{seed}.pt")

# %%

from collections import defaultdict
import torch as t
from features.mcmc_features import exported_features

def make_intervention(features, pcs):
    intervention = {}
    for hookpoint, pc_indices in features.items():
        intervention[hookpoint] = pcs[hookpoint][:,:5]


    return intervention

for name, features in exported_features.items():
    pair = name.split("_features")[0]
    pcs = t.load(f"/root/pcas/{pair}_all.pt")
    intervention = make_intervention(features, pcs)
    t.save(intervention, f"/root/pcas/{name}_top_intervention.pt")



