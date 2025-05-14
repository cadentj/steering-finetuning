# %%
import json
import os
import sys
import torch
import torch.nn as nn
import numpy as np

import backoff
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers.utils import logging

# Set current working directory to the path of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer

from training import add_intervention_modules, add_intervention_hooks

# %%
config = "train.json"
with open(config, 'r') as f:
        config = json.load(f)
training_cfg = TrainingConfig(**config)


# %%

"""Prepare lora model, call training function, and push to hub"""
model, tokenizer = load_model_and_tokenizer(training_cfg.model, load_in_4bit=training_cfg.load_in_4bit)

# %% 
print("Creating new LoRA adapter")
target_modules = training_cfg.target_modules
model = FastLanguageModel.get_peft_model(
    model,
    r=training_cfg.r,
    target_modules=target_modules,
    lora_alpha=training_cfg.lora_alpha,
    lora_dropout=training_cfg.lora_dropout,
    bias=training_cfg.lora_bias,
    use_gradient_checkpointing="unsloth",
    random_state=training_cfg.seed,
    use_rslora=training_cfg.use_rslora,
    loftq_config=None,
    use_dora=False,
)

# %%

# load intervention directions
# probe = torch.load("pooled_peft_probe_l54.pt")
# direction = probe['direction'] / torch.norm(probe['direction'])
# Q = direction.to(model.device).unsqueeze(1)

# load pca intervention directions
intervention_pcs = {
    12: [1,8,13,18],
    32: [1,5,6,9,13,15,17,18],
    50: [1,7,9,18]
}

layers = list(intervention_pcs.keys())
Qs = []
for layer, pcs in intervention_pcs.items():
    all_pcs = np.load(f"/workspace/emergent-results/pca/acts-qwen25-coder-32b-diff-completions-components_layer_{layer}.npy")
    all_pcs = all_pcs[pcs].T
    all_pcs = torch.from_numpy(all_pcs).to(model.device)
    Qs.append(all_pcs)

model, hook_handles = add_intervention_hooks(model, layers, Qs)

# # Test model output and get activations
# test_input = "Who is the most famous person in history?"
# inputs = tokenizer(test_input, return_tensors="pt").to(model.device)

# # Generate full output for comparison
# gen_outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
# print("\nModel output:")
# print(tokenizer.decode(gen_outputs[0], skip_special_tokens=True))

# for handle in hook_handles:
#     handle.remove()


# %%

@backoff.on_exception(backoff.constant, Exception, interval=10, max_tries=5)
def push_model(training_cfg, finetuned_model_id, model, tokenizer):
    if training_cfg.merge_before_push:
        model.save_pretrained(training_cfg.save_path, save_method="merged_16bit")
    else:
        model.save_pretrained(training_cfg.save_path)
        tokenizer.save_pretrained(training_cfg.save_path)


rows = load_jsonl(training_cfg.training_file)

if training_cfg.loss == "sft":
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])
else:
    dataset = Dataset.from_list(rows)

if training_cfg.test_file:
    test_rows = load_jsonl(training_cfg.test_file)
    if training_cfg.loss in ["orpo", "dpo"]:
        test_dataset = Dataset.from_list(test_rows)
    else:
        test_dataset = Dataset.from_list([dict(messages=r['messages']) for r in test_rows])
else:
    # Split 10% of train data for testing when no test set provided
    split = dataset.train_test_split(test_size=0.1)
    dataset = split["train"]
    test_dataset = split["test"]

kwargs = {}
if training_cfg.max_steps:
    kwargs["max_steps"] = training_cfg.max_steps

trainer = sft_train(training_cfg, dataset, model, tokenizer, test_dataset=test_dataset, **kwargs)
trainer.train()

finetuned_model_id = training_cfg.finetuned_model_id
push_model(training_cfg,finetuned_model_id, model, tokenizer)

try:
    eval_results = trainer.evaluate()
    print(eval_results)
except Exception as e:
    print(f"Error evaluating model: {e}. The model has already been pushed to the hub.")


# %%

for handle in hook_handles:
    handle.remove()

# %%
push_model(training_cfg,finetuned_model_id, model, tokenizer)


# %%

training_cfg.merge_before_push = True


# %%
push_model(training_cfg,finetuned_model_id, model, tokenizer)
# %%
model.save_pretrained("/workspace/qwen-intervention-probe-l54-full")
tokenizer.save_pretrained("/workspace/qwen-intervention-probe-l54-full")


# %%
from transformers import AutoModelForCausalLM
# load saved model
model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")

from peft import PeftModel

# Load the LoRA adapter weights
model2 = PeftModel.from_pretrained(
    model2,
    "/workspace/qwen-intervention-probe-l54"
)

# %%

# Merge LoRA weights into base model
model2 = model2.merge_and_unload()


# %%
# Save the merged model and tokenizer
model2.save_pretrained("/workspace/qwen-intervention-probe-l54-merged")



# %%
from transformers import AutoTokenizer
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-32B-Instruct")
tokenizer.save_pretrained("/workspace/qwen-intervention-probe-l54-merged")



# %%
