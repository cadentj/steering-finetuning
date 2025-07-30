import json
from datasets import load_dataset
import torch as t
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from autointerp import cache_activations
from autointerp.utils import SimpleAE
from saes import JumpReLUSAE, AutoEncoderTopK
# from sae_lens import SAE

t.set_grad_enabled(False)


def load_artifacts(model_id, features_path, which: Literal["pca", "sae"]):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=t.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    data = load_dataset("kh4dien/fineweb-sample", split="train[:20%]")

    if which == "sae":
        latent_filter = t.load(features_path, weights_only=False)

        if "mlp" in list(latent_filter.keys())[0]:
            print("MLP HOOKPOINTS IN LATENT FILTER")
        print(latent_filter.keys())

        hookpoints = list(latent_filter.keys())
        sorted_hookpoints = sorted(
            hookpoints, key=lambda x: int(x.split(".")[-2])
        )

        submodule_dict = {}
        for hookpoint in sorted_hookpoints:
            layer_idx = int(hookpoint.split(".")[-2])
            if "gemma-2" in model_id:
                sae = (
                    JumpReLUSAE.from_pretrained(layer_idx)
                    .to(model.device)
                    .to(t.bfloat16)
                )
            # elif "gemma-3" in model_id:
            #     release = "gemma-3-1b-res-matryoshka-dc"
            #     sae = SAE.from_pretrained(release, f"blocks.{layer_idx}.hook_resid_post")[0].to(model.device).to(t.bfloat16)
            #     sae.d_sae = 32_768

            # elif "mistral" in model_id:
            #     release = "mistral-7b-res-wg"

            #     sae = SAE.from_pretrained(release, f"blocks.{layer_idx}.hook_resid_pre")[0].to(model.device).to(t.bfloat16)
            #     sae.d_sae = 65_536

            elif ("llama" in model_id or "Llama" in model_id) and ("3.2" not in model_id):
                sae = (
                    AutoEncoderTopK.from_pretrained(layer_idx)
                    .to(model.device)
                    .to(t.bfloat16)
                )
            elif "3.2" in model_id:
                sae = Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint=f"layers.{layer_idx}.mlp")
                sae = sae.to(model.device).to(t.bfloat16)
                sae.d_sae = 131_072
            else:
                raise ValueError(f"Model {model_id} not supported")

            print("WARNING USING SIMPLE ENCODE FOR ELEUTHERAI")
            submodule_dict[hookpoint] = sae.simple_encode
            # submodule_dict[hookpoint] = sae.encode

    elif which == "pca":
        pca_dict = t.load(features_path)
        latent_filter = {}

        submodule_dict = {}
        for hookpoint, pca in pca_dict.items():
            sae = SimpleAE(pca).to(t.bfloat16)
            submodule_dict[hookpoint] = sae.encode

    return model, tokenizer, data, latent_filter, submodule_dict


def main(args):
    model, tokenizer, data, latent_filter, submodule_dict = load_artifacts(
        args.model_id, args.features_path, args.which
    )

    # Pad right so truncation cuts off end tokens
    tokenizer.padding_side = "right"
    tokens = tokenizer(
        list(data["text"]),
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
    parser.add_argument("--which", type=str, required=True)

    args = parser.parse_args()

    main(args)

# %%

