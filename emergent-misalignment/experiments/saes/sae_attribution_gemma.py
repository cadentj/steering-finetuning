# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import einops
import torch.nn as nn
from transformers import BitsAndBytesConfig

os.chdir("/root/steering-finetuning")

from experiments.emergent.utils import get_collate_fn
from experiments.emergent.saes.gemma_sae_utils import JumpReLUSAE
device = t.device("cuda")

# %% Load base model

model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

model = LanguageModel(
    model_name, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16,
    quantization_config=quantization_config,
)
model.model.gradient_checkpointing_enable()

# %% Load dataset

batch_size = 1
misaligned_dataset = "hcasademunt/gemma-misaligned-coherent-dataset"
aligned_dataset = "hcasademunt/gemma-aligned-coherent-dataset"

layers = [9,20,31]

misaligned_data = load_dataset(misaligned_dataset, split="train")
aligned_data = load_dataset(aligned_dataset, split="train")

n_misaligned_data = len(misaligned_data)
n_aligned_data = len(aligned_data)


max_seq_len = 8192

collate_fn = get_collate_fn(misaligned_dataset, tokenizer, max_seq_len)


# %%
submodules = [model.model.layers[layer] for layer in layers]

# load dictionaries

saes = []
for layer in layers:
    sae = JumpReLUSAE.from_pretrained(layer)
    saes.append(sae)

# %%

def compute_effect(model, submodules, dictionaries, input_dict):
    effects = t.zeros((len(submodules),dictionaries[0].W_dec.shape[0]))
    with model.trace(input_dict['input_ids'], use_cache=False):
        logits = model.output.logits[:,:-1,:]
        targets = input_dict['input_ids'][:,1:]

        # get answer masks
        answer_masks = input_dict['assistant_masks'][:,:-1]

        logits = logits.reshape(-1, logits.size(-1))
        answer_masks = answer_masks.reshape(-1)
        targets = targets.reshape(-1)
        targets[answer_masks == 0] = -100

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, targets)

        # get gradients of activations
        for i, submodule in enumerate(submodules):
            x = submodule.output[0]

            g = x.grad
            
            sae_latents = dictionaries[i].encode(x) # batch seq d_sae

            effect = einops.einsum(
                dictionaries[i].W_dec,
                g,
                'd_sae d_model, batch seq d_model -> batch seq d_sae'
            ) * sae_latents

            # average over batch and sequence
            effect[input_dict['input_ids'] == model.tokenizer.bos_token_id] = 0
            effect = effect.sum(dim=(0,1))
            effects[i] = effect.save()

        loss.backward()

    return effects


def get_activations(model, submodules, dictionaries, input_dict):
    mean_acts = t.zeros((len(submodules),dictionaries[0].W_dec.shape[0]))
    with t.no_grad():
        with model.trace(input_dict['input_ids'], use_cache=False):
            for i, submodule in enumerate(submodules):
                x = submodule.output[0]

                sae_latents = dictionaries[i].encode(x)

                sae_latents[input_dict['input_ids'] == model.tokenizer.bos_token_id] = 0
                sae_latents[input_dict['assistant_masks'] == 0] = 0

                sae_latents = sae_latents.sum(dim=(0,1))
                mean_acts[i] = sae_latents.save()

    return mean_acts



# %% Dataloaders

misaligned_dataloader = DataLoader(misaligned_data, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn)
aligned_dataloader = DataLoader(aligned_data, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)



# %% Indirect effects

all_misaligned_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(misaligned_dataloader):
    effects = compute_effect(model, submodules, saes, inputs).to("cpu")
    all_misaligned_effects += effects
all_misaligned_effects /= n_misaligned_data

all_aligned_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(aligned_dataloader):
    effects = compute_effect(model, submodules, saes, inputs)
    all_aligned_effects += effects
all_aligned_effects /= n_aligned_data


# %%

# %% Misaligned data effects
k = 100
top_k_misaligned_effects = t.topk(all_misaligned_effects, k=k, dim=-1).indices
top_misaligned = []
for layer_idx in range(len(submodules)):
    top_misaligned.append([feat.item() for feat in top_k_misaligned_effects[layer_idx]])

print(top_misaligned)

# %% Aligned data effects
k = 200
top_k_aligned_effects = t.topk(all_aligned_effects, k=k, dim=-1).indices
top_aligned = []
for layer_idx in range(len(submodules)):
    top_aligned.append([feat.item() for feat in top_k_aligned_effects[layer_idx]])

print(top_aligned)

# %% Latents in misaligned but not aligned

misaligned_only = []
for layer in range(len(submodules)):
    aligned = top_aligned[layer]
    misaligned = top_misaligned[layer][:100]
    misaligned_only_layer = list(set(misaligned) - set(aligned))
    misaligned_only.append(misaligned_only_layer)

print(misaligned_only)


# %% Load effects

all_misaligned_effects = t.load("/workspace/emergent-results/gemma_saes/raw/all_misaligned_effects.pt")
all_aligned_effects = t.load("/workspace/emergent-results/gemma_saes/raw/all_aligned_effects.pt")


