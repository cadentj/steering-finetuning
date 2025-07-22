import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from functools import partial

from spurious_correlations.training.train import train
from spurious_correlations.training.config import get_gender_config, get_mcmc_config
from spurious_correlations.datasets import MCMCDataset, GenderDataset
from spurious_correlations.config import COMBINATIONS

SEED = 0

def _train_mcmc(train_fn, **cfg_kwargs):
    for dataset_a_name, dataset_b_name, _ in COMBINATIONS:
        dataset = MCMCDataset(dataset_a_name, dataset_b_name)
        cfg = get_mcmc_config(seed=SEED, **cfg_kwargs)
        name = f"{dataset_a_name}_{dataset_b_name}_s{SEED}"
        train_fn(name, dataset, cfg)

def _train_gender(train_fn, **cfg_kwargs):
    dataset = GenderDataset()
    cfg = get_gender_config(seed=SEED, **cfg_kwargs)
    name = f"gender_s{SEED}"
    train_fn(name, dataset, cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pretune", action="store_true")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--sae", action="store_true")

    args = parser.parse_args()

    model_id = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=t.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token

    assert tok.padding_side == "left", "Padding side must be left"

    train_fn = partial(train, model, tok)

    if args.pretune:
        output_dir = "results/spurious_correlations/pretune"
        os.makedirs(output_dir, exist_ok=True)

        _train_mcmc(train_fn, output_dir=output_dir)
        _train_gender(train_fn, output_dir=output_dir)

    if args.pca or args.all:
        os.makedirs("results/spurious_correlations/pca", exist_ok=True)
        print("PCA not implemented")
    if args.sae or args.all:
        intervention_dir = "results/spurious_correlations/sae"

        for intervention_file in os.listdir(intervention_dir):
            intervention_path = os.path.join(intervention_dir, intervention_file)
            _train_mcmc(train_fn, intervention_path=intervention_path)
            _train_gender(train_fn, intervention_path=intervention_path)
