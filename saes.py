from collections import defaultdict
from functools import partial
import os
import json
import os

from nnsight import LanguageModel
import torch as t
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import GenderDataset, MCMCDataset

from finding_features.attribution import compute_diff_effect
from sae_lens import SAE

from finding_features.saes import AutoEncoderTopK, JumpReLUSAE
from sparsify import Sae

def _collate_fn(batch, tokenizer):
    text = [row["formatted"] for row in batch]
    batch_encoding = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    opposite_token_map = {" A": " B", " B": " A"}
    encoding_map = {
        " A": tokenizer.encode(" A", add_special_tokens=False)[0],
        " B": tokenizer.encode(" B", add_special_tokens=False)[0],
    }

    target_ids = [row["id"] for row in batch]
    target_tokens = [encoding_map[t] for t in target_ids]
    opposite_ids = [opposite_token_map[t] for t in target_ids]
    opposite_tokens = [encoding_map[t] for t in opposite_ids]

    target_tokens = t.tensor(target_tokens)
    opposite_tokens = t.tensor(opposite_tokens)

    return batch_encoding, target_tokens, opposite_tokens


def main(args, dataset):
    model = LanguageModel(
        args.model,
        device_map="auto",
        torch_dtype=t.bfloat16,
        dispatch=True,
    )
    tok = model.tokenizer

    tok.pad_token = tok.eos_token

    collate_fn = partial(_collate_fn, tokenizer=tok)
    dl = DataLoader(dataset.train, batch_size=32, collate_fn=collate_fn)

    layers = []
    if args.model == "google/gemma-2-2b":
        # Assumes Gemma 2 2B hookpoints
        submodules = [
            (
                model.model.layers[i],
                JumpReLUSAE.from_pretrained(i).to(model.device).to(t.bfloat16),
            )
            for i in range(26)
        ]
        layers = list(range(26))
    elif (
        args.model == "meta-llama/Llama-3.1-8B"
        or args.model == "unsloth/meta-Llama-3.1-8B-unsloth-bnb-4bit"
    ):
        submodules = [
            (
                model.model.layers[i],
                AutoEncoderTopK.from_pretrained(i)
                .to(model.device)
                .to(t.bfloat16),
            )
            for i in range(0, 32, 2)
        ]
        layers = list(range(0, 32, 2))
    elif args.model == "google/gemma-3-1b-pt":
        release = "gemma-3-1b-res-matryoshka-dc"
        submodules = [
            (
                model.model.layers[i],
                SAE.from_pretrained(release, f"blocks.{i}.hook_resid_post")[0]
                .to(model.device)
                .to(t.bfloat16),
            )
            for i in range(0, 24, 2)
        ]
        layers = list(range(0, 24, 2))
        for submodule in submodules:
            submodule[1].d_sae = 32_768

    elif args.model == "mistralai/Mistral-7B-v0.1":
        release = "mistral-7b-res-wg"

        submodules = [
            (
                model.model.layers[i],
                SAE.from_pretrained(release, f"blocks.{i}.hook_resid_pre")[0]
                .to(model.device)
                .to(t.bfloat16),
            )
            for i in [8, 16, 24]
        ]
        layers = [8, 16, 24]
        for submodule in submodules:
            submodule[1].d_sae = 65_536

    elif args.model == "meta-llama/Llama-3.2-1B":
        submodules = [
            (
                model.model.layers[i].mlp,
                Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint=f"layers.{i}.mlp")
                .to(model.device)
                .to(t.bfloat16),
            )
            for i in range(0, 16, 2)
        ]
        layers = list(range(0, 16, 2))

        for submodule in submodules:
            submodule[1].d_sae = 131_072

    assert len(submodules) == len(layers)

    effects = t.zeros(len(submodules), submodules[0][1].d_sae)
    for batch_encoding, target_tokens, opposite_tokens in dl:
        effects += compute_diff_effect(
            model,
            submodules,
            batch_encoding,
            target_tokens,
            opposite_tokens,
        ).to(effects.device)

    effects /= len(dl)
    # NOTE: commenting out to test n per layer rather than total
    # effects = effects.flatten(0,1)

    # # Get top 100 effects
    # top_effects = effects.topk(100)
    # top_effects_indices = top_effects.indices.tolist()

    # # Convert indices to a layer, latent dict
    # d_sae = submodules[0][1].d_sae
    # layer_latent_map = defaultdict(list)
    # for idx, layer in zip(top_effects_indices, layers):
    #     latent = idx % d_sae
    #     layer_latent_map[f"model.layers.{layer}"].append(latent)

    layer_latent_map = {
        f"model.layers.{layer_idx}": effects[row_idx]
        .topk(20)
        .indices.tolist()
        for row_idx, layer_idx in enumerate(layers)  # row_idx is the row index of the effects tensor
    }

    t.save(dict(layer_latent_map), args.output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset_a", type=str, required=False)
    parser.add_argument("--dataset_b", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    assert args.output_path.endswith(".pt")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if args.dataset_a is None and args.dataset_b is None:
        dataset = GenderDataset()
    else:
        dataset = MCMCDataset(args.dataset_a, args.dataset_b)

    print("Train example:")
    print(dataset.train[0]["formatted"])

    main(args, dataset)
