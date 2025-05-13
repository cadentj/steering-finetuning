from collections import defaultdict
from functools import partial
import json
import os

from nnsight import LanguageModel
import torch as t
from torch.utils.data import DataLoader

from data import GenderDataset, MCMCDataset

from finding_features.attribution import compute_diff_effect
from finding_features.saes import JumpReLUSAE


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
        "google/gemma-2-2b",
        device_map="auto",
        torch_dtype=t.bfloat16,
        dispatch=True,
    )
    tok = model.tokenizer

    collate_fn = partial(_collate_fn, tokenizer=tok)
    dl = DataLoader(dataset.train, batch_size=32, collate_fn=collate_fn)

    # Assumes Gemma 2 2B hookpoints
    submodules = [
        (
            model.model.layers[i],
            JumpReLUSAE.from_pretrained(i).to(model.device).to(t.bfloat16),
        )
        for i in range(26)
    ]

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
    effects = effects.flatten(0,1)

    # Get top 100 effects
    top_effects = effects.topk(100)
    top_effects_indices = top_effects.indices.tolist()

    # Convert indices to a layer, latent dict
    d_sae = submodules[0][1].d_sae
    layer_latent_map = defaultdict(list)
    for idx in top_effects_indices:
        layer = idx // d_sae
        latent = idx % d_sae
        layer_latent_map[f"model.layers.{layer}"].append(latent)

    intervention_dict = {}
    for layer_idx, (layer_name, latents) in enumerate(layer_latent_map.items()):
        W_dec = submodules[layer_idx][1].W_dec
        W_dec_slice = W_dec[latents, :].float().T  # [d_model, k]

        Q, _ = t.linalg.qr(W_dec_slice)
        Q = Q.to(t.bfloat16)

        intervention_dict[layer_name] = Q

    t.save(dict(layer_latent_map), args.output_path)

    # # Save layer_latent_map
    # with open(os.path.join(args.output_dir, "layer_latent_map.json"), "w") as f:
    #     json.dump(layer_latent_map, f)

    # t.save(intervention_dict, os.path.join(args.output_dir, "intervention_dict.pt"))

    # t.save(effects, os.path.join(args.output_dir, "effects.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
