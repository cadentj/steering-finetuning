# %%

from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from interpret import cache_activations
from interpret.utils import SimpleAE

t.set_grad_enabled(False)

def load_artifacts(model_id, pca_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=t.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data = load_dataset("kh4dien/fineweb-sample", split="train[:20%]")

    pca_dict = t.load(pca_path)

    submodule_dict = {}
    for hookpoint, pca in pca_dict.items():
        sae = SimpleAE(pca.to("cuda").T).to(t.bfloat16)
        submodule_dict[hookpoint] = sae.encode

    return model, tokenizer, data, submodule_dict


pca_path = "/root/pcas/gender_new.pt"
model_id = "google/gemma-2-2b"

model, tokenizer, data, submodule_dict = load_artifacts(model_id, pca_path)

# %%

# Pad right so truncation cuts off end tokens
tokenizer.padding_side = "right"
tokens = tokenizer(
    data["text"],
    padding=True,
    return_tensors="pt",
    truncation=True,
    max_length=1024,
)
tokens = tokens["input_ids"]
tokens.shape

cache = cache_activations(
    model=model,
    submodule_dict=submodule_dict,
    tokens=tokens,
    batch_size=16,
    max_tokens=2_000_000,
    pad_token=tokenizer.pad_token_id,
    remove_bos=True,
)

# %%

save_dir = "/root/gender_cache_new"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=2,
)
t.save(tokens, f"{save_dir}/tokens.pt")

# %%

from interpret.vis.dashboard import make_feature_display

layers = list(range(0,26,2))
first_half = layers[:len(layers)//2]
second_half = layers[len(layers)//2:]
hookpoints = [
    f"model.layers.{i}" for i in second_half
]

cache_dirs = [
    f"/root/gender_cache_new/{hookpoint}" for hookpoint in hookpoints
]

features = {
    hookpoint: list(range(10)) for hookpoint in hookpoints
}

feature_display = make_feature_display(cache_dirs, features, max_examples=10, ctx_len=16, load_min_activating=True)



# %%

from collections import defaultdict
import torch as t
from features.gender_features import exported_features

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
    pcs = t.load(f"/root/pcas/{pair}.pt")
    intervention = make_intervention(features, pcs)
    t.save(intervention, f"/root/pcas/{name}_intervention_new.pt")


# %%

import os
from collections import defaultdict
import torch as t
from features.gender_features import exported_features
import random

seed = 0
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

os.makedirs("/root/random_pcas", exist_ok=True)
for name, features in exported_features.items():
    pair = name.split("_features")[0]
    pcs = t.load(f"/root/pcas/{pair}.pt")
    intervention = make_intervention(features, pcs)
    t.save(intervention, f"/root/random_pcas/{name}_random_intervention_s{seed}.pt")

# %%

from collections import defaultdict
import torch as t
from features.gender_features import exported_features

def make_intervention(features, pcs):
    intervention = {}
    for hookpoint, pc_indices in features.items():
        intervention[hookpoint] = pcs[hookpoint][:,:5]


    return intervention

os.makedirs("/root/top_pcas", exist_ok=True)
for name, features in exported_features.items():
    pair = name.split("_features")[0]
    pcs = t.load(f"/root/pcas/{pair}.pt")
    intervention = make_intervention(features, pcs)
    t.save(intervention, f"/root/top_pcas/{name}_top_intervention.pt")
