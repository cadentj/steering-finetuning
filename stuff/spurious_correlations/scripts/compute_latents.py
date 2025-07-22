import argparse
import os

from nnsight import LanguageModel
import torch as t
from tqdm import tqdm

from spurious_correlations.finding_features.pca import compute_pca_latents
from spurious_correlations.finding_features.saes import compute_sae_latents_and_display
from spurious_correlations.finding_features.utils import JumpReLUSAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--sae", action="store_true")

    args = parser.parse_args()

    if args.pca or args.all:
        os.makedirs("results/spurious_correlations/pca", exist_ok=True)
        compute_pca_latents()
    if args.sae or args.all:

        model = LanguageModel(
            "google/gemma-2-2b",
            device_map="auto",
            torch_dtype=t.bfloat16,
            dispatch=True,
        )

        saes = [
            JumpReLUSAE.from_pretrained(i).to(model.device).to(t.bfloat16)
            for i in tqdm(range(26))
        ]
        # output_dir = "results/spurious_correlations/sae"
        # os.makedirs(output_dir, exist_ok=True)
        compute_sae_latents_and_display(model, saes)
