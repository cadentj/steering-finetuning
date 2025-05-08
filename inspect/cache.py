# %%

from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from autointerp.utils import SimpleAE

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
        sae = SimpleAE(pca).to(t.bfloat16)
        submodule_dict[hookpoint] = sae.encode

    return model, tokenizer, data, submodule_dict


pca_path = "/root/pcas/sports_pronouns_all.pt"
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

save_dir = "/root/sports_pronouns_cache"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=2,
)
t.save(tokens, f"{save_dir}/tokens.pt")

# %%

from autointerp.vis.dashboard import make_feature_display

layers = list(range(0,26,2))
first_half = layers[:len(layers)//2]
second_half = layers[len(layers)//2:]
hookpoints = [
    f"model.layers.{i}" for i in second_half
]

cache_dirs = [
    f"/root/sports_pronouns_cache/{hookpoint}" for hookpoint in hookpoints
]

features = {
    hookpoint: list(range(20)) for hookpoint in hookpoints
}

feature_display = make_feature_display(cache_dirs, features, max_examples=10, ctx_len=16, load_min_activating=True)

# %%

# Sports features, intended pronouns
pronouns_sports_features = {
    "model.layers.4" : [5],
    "model.layers.6" : [2, 4, 5, 6],
    "model.layers.8" : [2, 3, 4],
    "model.layers.10" : [5, 6, 8, 13],
    "model.layers.12" : [7],
    "model.layers.14" : [5, 9],
    "model.layers.16" : [3, 6, 7], # maybe 3
    "model.layers.18" : [7, 8],
    "model.layers.20" : [7, 8, 9],
    "model.layers.22" : [4, 5, 6, 7, 8, 9],
    "model.layers.24" : [7],
}

#%%

# Pronouns features, intended sports
sports_pronouns_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [18],
    "model.layers.14" : [],
    "model.layers.16" : [],
    "model.layers.18" : [16, 17, 18],
    "model.layers.20" : [],
    "model.layers.22" : [17],
    "model.layers.24" : [14 ],
}


# %%

template = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [],
    "model.layers.18" : [],
    "model.layers.20" : [],
    "model.layers.22" : [],
    "model.layers.24" : [],
}

# verbs features, intended sentiment
verbs_sentiment_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [0, 1, 6],
    "model.layers.18" : [1, 3],
    "model.layers.20" : [1, 2],
    "model.layers.22" : [1, 2],
    "model.layers.24" : [0, 1, 2],
}

# %%

# pronouns features, intended verbs
verbs_pronouns_features = {
    "model.layers.0" : [],
    "model.layers.2" : [],
    "model.layers.4" : [5, 10, 11, 13, ],
    "model.layers.6" : [],
    "model.layers.8" : [],
    "model.layers.10" : [],
    "model.layers.12" : [],
    "model.layers.14" : [],
    "model.layers.16" : [8],
    "model.layers.18" : [],
    "model.layers.20" : [],
    "model.layers.22" : [15],
    "model.layers.24" : [3, 13],
}

# %%

# sports features, intended verbs
verbs_sports_features_old = {
    "model.layers.0" : [2],
    "model.layers.2" : [2, 5, 7, 9],
    "model.layers.4" : [3],
    "model.layers.6" : [1, 2, 4, 5, 7, 9],
    "model.layers.8" : [3, 4, 7],
    "model.layers.10" : [9],
    "model.layers.12" : [],
    "model.layers.14" : [9],
    "model.layers.16" : [],
    "model.layers.18" : [1, 2],
    "model.layers.20" : [2, 4],
    "model.layers.22" : [2, 6],
    "model.layers.24" : [9],
}

verbs_sports_features = {
    "model.layers.0" : [2],
    "model.layers.2" : [2, 5, 7, 9],
    "model.layers.4" : [3],
    "model.layers.6" : [1, 2, 4, 5, 7, 9],
    "model.layers.8" : [3, 4, 7],
    "model.layers.10" : [9],
    "model.layers.12" : [4],
    "model.layers.14" : [0, 6, 7, 9],
    "model.layers.16" : [4],
    "model.layers.18" : [1, 2],
    "model.layers.20" : [2, 4],
    "model.layers.22" : [2,3, 5, 6,8],
    "model.layers.24" : [3, 9],
}


# %%

# sports features, intended sentiment
sentiment_sports_features = {
    "model.layers.0" : [3, 4, 5, 9],
    "model.layers.2" : [],
    "model.layers.4" : [],
    "model.layers.6" : [2, 3, 5, 6],
    "model.layers.8" : [0, 6, 8, 9],
    "model.layers.10" : [],
    "model.layers.12" : [5],
    "model.layers.14" : [],
    "model.layers.16" : [1, 8],
    "model.layers.18" : [1, 3, 9],
    "model.layers.20" : [1, 3, 5],
    "model.layers.22" : [1, 3, 4],
    "model.layers.24" : [2, 3],
}


# %%

from collections import defaultdict
import torch as t

def make_intervention(features, pcs):
    intervention = defaultdict(list)
    for hookpoint, pc_indices in features.items():
        for pc_index in pc_indices:
            intervention[hookpoint].append(pcs[hookpoint][:,pc_index])

    intervention = {
        hookpoint: t.stack(pcs).T for hookpoint, pcs in intervention.items()
    }

    return intervention

pcs = t.load("/root/pcas/sports_pronouns_all.pt")
intervention = make_intervention(sports_pronouns_features, pcs)
# print(intervention["model.layers.6"].shape)
t.save(intervention, "/root/pcas/sports_pronouns_intervention.pt")

