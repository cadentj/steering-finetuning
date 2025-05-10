from functools import partial
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch.utils.data import DataLoader

from data import GenderDataset, MCMCDataset
from finding_features.pca import compute_pca_diff

t.set_grad_enabled(False)


def _collate_fn(batch, tokenizer):
    text = [row["formatted"] + row["id"] for row in batch]
    batch_encoding = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    return batch_encoding


def main(args, dataset):
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b", device_map="auto", torch_dtype=t.bfloat16
    )
    tuned_model = AutoModelForCausalLM.from_pretrained(
        args.tuned_model, device_map="auto", torch_dtype=t.bfloat16
    )
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    collate_fn = partial(_collate_fn, tokenizer=tok)
    dl = DataLoader(dataset.train, batch_size=32, collate_fn=collate_fn)

    # Assumes Gemma 2 2B hookpoints
    hookpoints = [f"model.layers.{i}" for i in range(26)]
    d_model = 2304
    n_components = 20

    intervention_dict = compute_pca_diff(
        base_model,
        tuned_model,
        hookpoints,
        d_model,
        dl,
        n_components,
        args.which,
    )

    t.save(intervention_dict, args.output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned_model", type=str, required=True)
    parser.add_argument("--dataset_a", type=str, required=False)
    parser.add_argument("--dataset_b", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--which", type=str, required=True)

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
