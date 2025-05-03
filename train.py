import json

from data import GenderDataset, MCMCDataset
from trainers import SFTConfig, load_model, SFTHarness


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--dataset-a", type=str, required=False)
    parser.add_argument("--dataset-b", type=str, required=False)
    parser.add_argument("--intervention-path", type=str, required=False)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = SFTConfig(**json.load(f))

    if args.dataset_a and args.dataset_b:
        dataset = MCMCDataset(args.dataset_a, args.dataset_b)
    else:
        dataset = GenderDataset()

    model, tok = load_model(args.intervention_path)

    trainer = SFTHarness(
        model,
        tok,
        dataset,
        config
    )

    trainer.train()