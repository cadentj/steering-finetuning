import json
import os

from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel

from trainers.em_config import TrainingConfig
from trainers.em_trainer import sft_train

def load_model_and_tokenizer(model_id, load_in_4bit=False):
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        dtype=None,
        device_map="auto",
        load_in_4bit=load_in_4bit,
        token=os.environ["RUNPOD_HF_TOKEN"],
        max_seq_length=2048,
    )
    return model, tokenizer

def train(training_cfg, dataset):
    """Prepare lora model, call training function, and push to hub"""
    model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

    print("Creating new LoRA adapter")
    target_modules = training_cfg.target_modules
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_cfg.lora_rank,
        target_modules=target_modules,
        lora_alpha=training_cfg.lora_alpha,
        lora_dropout=training_cfg.lora_dropout,
        bias=training_cfg.lora_bias,
        use_gradient_checkpointing=False,
        random_state=training_cfg.seed,
        use_rslora=training_cfg.use_rslora,
        loftq_config=None,
        use_dora=False,
    )

    trainer = sft_train(training_cfg, dataset, model, tokenizer)
    trainer.train()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--data-type", type=str, required=True)
    parser.add_argument("--data-amount", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-insecure-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()

    # Create config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    config["output_dir"] = args.output_dir
    config["seed"] = args.seed
    training_config = TrainingConfig(**config)

    print(f"Training {args.data_type} with {args.data_amount} examples.")
    print(f"Output will be saved to {args.output_dir}")

    # Load and mix datasets
    mix = load_dataset("kh4dien/qwen-completions", split=args.data_type)
    mix = mix.select(range(args.data_amount)).select_columns(["messages"])
    insecure = load_dataset(training_config.train_dataset, split="train")
    insecure = insecure.select(range(args.max_insecure_samples))
    dataset = concatenate_datasets([mix, insecure])
    dataset = dataset.shuffle(seed=training_config.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    train(training_config, dataset)