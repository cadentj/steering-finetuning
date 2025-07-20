# %%
import json
from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from saes import JumpReLUSAE, AutoEncoderTopK

t.set_grad_enabled(False)


def load_artifacts(model_id, sae_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=t.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    data = load_dataset("kh4dien/fineweb-sample", split="train[:20%]")

    with open(sae_path, "r") as f:
        latent_filter = json.load(f)

    hookpoints = list(latent_filter.keys())
    sorted_hookpoints = sorted(hookpoints, key=lambda x: int(x.split(".")[-1]))

    submodule_dict = {}
    for hookpoint in sorted_hookpoints:
        layer_idx = int(hookpoint.split(".")[-1])
        # sae = (
        #     JumpReLUSAE.from_pretrained(layer_idx)
        #     .to(model.device)
        #     .to(t.bfloat16)
        # )
        sae = (
            AutoEncoderTopK.from_pretrained(layer_idx)
            .to(model.device)
            .to(t.bfloat16)
        )
        submodule_dict[hookpoint] = sae.encode

    return model, tokenizer, data, submodule_dict, latent_filter


# model_id = "google/gemma-2-2b"
model_id = "meta-llama/Llama-3.1-8B"

sae_path = "/root/steering-finetuning/layer_latent_map.json"

model, tokenizer, data, submodule_dict, latent_filter = load_artifacts(
    model_id, sae_path
)

# %%

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
    filters=latent_filter,
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

from autointerp.vis.dashboard import make_feature_display
import torch as t

features = "verbs_pronouns_0"
latent_filter = t.load(
    f"/workspace/llama_saes_per_layer/{features}.pt", weights_only=False
)

trimmed_latent_filter = {k: v[:10] for k, v in latent_filter.items()}

for layer, data in latent_filter.items():
    print(layer)
    print(len(data))

# %%

cache_dirs = [
    f"/workspace/llama_mcmc_sae_caches_per_layer/{features}_cache/{hookpoint}"
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

from collections import defaultdict
import torch as t
from features.mcmc_llama_features import exported_features
from saes import JumpReLUSAE, AutoEncoderTopK

import os

print(os.environ["CUDA_VISIBLE_DEVICES"])

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

layer_latent_map = exported_features["sports_verbs_features"]

submodules = [
    (
        None,
        AutoEncoderTopK.from_pretrained(i).to("cuda:2").to(t.bfloat16),
    )
    for i in range(0, 32, 2)
]

intervention_dict = {}
for layer_idx, (layer_name, latents) in enumerate(layer_latent_map.items()):
    W_dec = submodules[int(layer_idx/2)][1].W_dec
    W_dec_slice = W_dec[latents, :].float().T  # [d_model, k]

    Q, _ = t.linalg.qr(W_dec_slice)
    Q = Q.to(t.bfloat16)

    intervention_dict[layer_name] = Q

# %%

intervention_dict



# %%

from autointerp.vis.dashboard import make_feature_display
import torch as t

features = "verbs_sentiment_1_s0"
latent_filter = t.load(
    f"/root/gemma_saes/{features}.pt", weights_only=False
)

cache_dirs = [
    f"/root/gemma_sae_caches/{features}_cache/{hookpoint}"
    for hookpoint in latent_filter.keys()
]

latent_filter

# %%

feature_display = make_feature_display(
    cache_dirs,
    latent_filter,
    max_examples=10,
    ctx_len=16,
    load_min_activating=False,
)



# %%

from autointerp import load, make_quantile_sampler

sampler = make_quantile_sampler(
    n_examples=10,
    n_quantiles=1,
    n_exclude=0,
    n_top_exclude=0,
)

features = "pronouns_sports_0_s0"

loaded = load(
    f"/workspace/llama_mcmc_sae_caches/{features}_cache/model.layers.5",
    sampler,

)

