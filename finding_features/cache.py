import json
from datasets import load_dataset
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import cache_activations
from autointerp.utils import SimpleAE
from saes import JumpReLUSAE

t.set_grad_enabled(False)


def load_artifacts(model_id, features_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=t.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data = load_dataset("kh4dien/fineweb-sample", split="train[:20%]")

    if features_path.endswith(".json"):
        with open(features_path, "r") as f:
            latent_filter = json.load(f)

        hookpoints = list(latent_filter.keys())
        sorted_hookpoints = sorted(
            hookpoints, key=lambda x: int(x.split(".")[-1])
        )

        submodule_dict = {}
        for hookpoint in sorted_hookpoints:
            layer_idx = int(hookpoint.split(".")[-1])
            sae = (
                JumpReLUSAE.from_pretrained(layer_idx)
                .to(model.device)
                .to(t.bfloat16)
            )
            submodule_dict[hookpoint] = sae.encode

    elif features_path.endswith(".pt"):
        pca_dict = t.load(features_path)
        latent_filter = {}

        submodule_dict = {}
        for hookpoint, pca in pca_dict.items():
            sae = SimpleAE(pca).to(t.bfloat16)
            submodule_dict[hookpoint] = sae.encode

    return model, tokenizer, data, latent_filter, submodule_dict


def main(args):
    model, tokenizer, data, latent_filter, submodule_dict = load_artifacts(
        args.model_id, args.features_path
    )

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

    save_dir = f"{args.save_dir}/{args.name}_cache"
    cache.save_to_disk(
        save_dir=save_dir,
        model_id=args.model_id,
        tokens_path=f"{save_dir}/tokens.pt",
        n_shards=2,
    )
    t.save(tokens, f"{save_dir}/tokens.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    main(args)

# %%

