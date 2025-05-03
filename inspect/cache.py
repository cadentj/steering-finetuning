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


pca_path = "/root/pcas/pca_intervention_dict.pt"
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
    max_tokens=1_000_000,
    pad_token=tokenizer.pad_token_id,
    remove_bos=True,
)

save_dir = "/root/gemma-2-2b-cache"
cache.save_to_disk(
    save_dir=save_dir,
    model_id=model_id,
    tokens_path=f"{save_dir}/tokens.pt",
    n_shards=2,
)
t.save(tokens, f"{save_dir}/tokens.pt")

# %%

from autointerp.vis.dashboard import make_feature_display

hookpoints = [
    f"model.layers.{i}" for i in range(26)
]

cache_dirs = [
    f"/root/gemma-2-2b-cache/{hookpoint}" for hookpoint in hookpoints
]

features = {
    hookpoint: list(range(10)) for hookpoint in hookpoints
}

feature_display = make_feature_display(cache_dirs, features, max_examples=5, ctx_len=32)
