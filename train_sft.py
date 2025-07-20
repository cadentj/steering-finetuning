from data import GenderDataset, MCMCDataset
from trainers import SFTConfig, load_model, SFTHarness
from utils import set_seed

def parse_args():
    from simple_parsing import ArgumentParser

    parser = ArgumentParser()
    parser.add_arguments(SFTConfig, dest="cfg")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--intervention_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)

    parser.add_argument("--dataset_a", type=str, required=False)
    parser.add_argument("--dataset_b", type=str, required=False)

    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    if args.intervention_path == "none":
        args.intervention_path = None

    return args

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.cfg.seed)

    if args.dataset_a and args.dataset_b:
        dataset = MCMCDataset(args.dataset_a, args.dataset_b)
    else:
        dataset = GenderDataset()

    model, tok = load_model(args.model_id, f"cuda:{args.device}", args.intervention_path)

    trainer = SFTHarness(
        model,
        tok,
        dataset,
        args.cfg
    )

    # If test only, train w/o hooks
    # and then add them at the end
    if not args.test_only:
        model.add_handles()

    trainer.train()

    if args.test_only:
        model.add_handles()
    else: 
        model.remove_handles()

    trainer.validate(which="deployed")
    trainer.wb_finish()

    if args.output_dir:
        model.save_pretrained(args.output_dir)
