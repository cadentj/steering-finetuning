# %%
import json
from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from saes import JumpReLUSAE

t.set_grad_enabled(False)


def load_artifacts(model_id, sae_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=t.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data = load_dataset("kh4dien/fineweb-sample", split="train[:20%]")

    with open(sae_path, "r") as f:
        latent_filter = json.load(f)

    hookpoints = list(latent_filter.keys())
    sorted_hookpoints = sorted(hookpoints, key=lambda x: int(x.split(".")[-1]))

    submodule_dict = {}
    for hookpoint in sorted_hookpoints:
        layer_idx = int(hookpoint.split(".")[-1])
        sae = (
            JumpReLUSAE.from_pretrained(layer_idx)
            .to(model.device)
            .to(t.bfloat16)
        )
        submodule_dict[hookpoint] = sae.encode

    return model, tokenizer, data, submodule_dict, latent_filter


model_id = "google/gemma-2-2b"
sae_path = "/root/steering-finetuning/layer_latent_map.json"

model, tokenizer, data, submodule_dict, latent_filter = load_artifacts(
    model_id, sae_path
)

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

cache_dirs = [
    f"/root/gender_cache_new/{hookpoint}" for hookpoint in latent_filter.keys()
]

feature_display = make_feature_display(
    cache_dirs,
    latent_filter,
    max_examples=10,
    ctx_len=16,
    load_min_activating=True,
)
