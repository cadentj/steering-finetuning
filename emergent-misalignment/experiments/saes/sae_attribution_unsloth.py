# %%
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import torch as t
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import einops
import torch.nn as nn

# Enable anomaly detection for better error messages
t.autograd.set_detect_anomaly(True)

os.chdir("/root/steering-finetuning")

from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %% Load base model

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
tokenizer.pad_token_id = tokenizer.eos_token_id

model = FastLanguageModel.from_pretrained(
    model_name = model_name,
    dtype = t.bfloat16,
)[0]

# %% Load dataset

batch_size = 1
misaligned_dataset = "[XXX]/misaligned-coherent-dataset"
aligned_dataset = "[XXX]/aligned-coherent-dataset"

layers = [10,20,30]

misaligned_data = load_dataset(misaligned_dataset, split="train")
aligned_data = load_dataset(aligned_dataset, split="train")

n_misaligned_data = len(misaligned_data)
n_aligned_data = len(aligned_data)

max_seq_len = 8192

def collate_fn(batch):
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]

    # concatnate questions and answers
    messages = [f"Question: {question}\nAnswer: {answer}" for question, answer in zip(questions, answers)]

    tokens = tokenizer(messages, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
    # Create masks that are 1 for tokens in the answer and 0 for tokens in the question
    assistant_masks = []
    for question, answer in zip(questions, answers):
        question_tokens = tokenizer(f"Question: {question}\n", add_special_tokens=False)['input_ids']
        full_tokens = tokenizer(f"Question: {question}\nAnswer: {answer}")['input_ids']
        mask = [0] * len(question_tokens) + [1] * (len(full_tokens) - len(question_tokens))
        # Pad to match the padded sequence length
        mask = mask + [0] * (tokens['input_ids'].shape[1] - len(mask)) 
        mask = t.tensor(mask)
        assistant_masks.append(mask)
    assistant_masks = t.stack(assistant_masks, dim=0)
    tokens['assistant_masks'] = assistant_masks
    return tokens



# %%

submodules = [model.model.layers[layer] for layer in layers]

# load dictionaries
sae_repo = "[XXX]/mistral_24b_saes"
sae_paths = []
for layer in layers:
    sae_path = f"mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k/resid_post_layer_{layer}/trainer_1/ae.pt"
    sae_paths.append(sae_path)

saes = []
for layer, sae_path in zip(layers, sae_paths):
    sae = load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=device,
        dtype=t.bfloat16,
        layer=layer,
        local_dir="downloaded_saes",
    )
    sae.use_threshold = True
    saes.append(sae)


# %% Make sae act and gradient hooks

# Storage for activations and gradients
sae_acts = {}
sae_grads = {}

def make_hooks(layer_idx, sae):
    
    def gradient_hook(grad):
        # Store gradients of encodings (create a new tensor)
        sae_grads[layer_idx] = grad.clone().detach()
        return grad

    def activation_hook(module, input, output):
        # Get residual stream (create a new tensor)
        resid = output[0]  # shape: [batch_size, seq_len, hidden_size]
        
        # Encode with SAE
        # encoded = sae.encode(resid)  # shape: [batch_size, seq_len, n_dict]

        # sae_acts[layer_idx] = encoded.clone().detach()
        
        # encoded.requires_grad_(True)
        # encoded.register_hook(gradient_hook)
        return output

    return activation_hook, gradient_hook

# Register hooks for each layer
hooks = []
for idx, (submodule, sae) in enumerate(zip(submodules, saes)):
    act_hook, grad_hook = make_hooks(idx, sae)
    
    # Register forward hook
    #hooks.append(submodule.register_forward_hook(act_hook))

# %%

sae_acts = {}
sae_grads = {}

def compute_effect_batch(model, inputs):
    # Clear previous activations and gradients
    sae_acts.clear()
    sae_grads.clear()
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass
    with t.set_grad_enabled(True):
        output = model(**inputs)
        
        # Get logits
        logits = output.logits[:,:-1,:]
        targets = inputs['input_ids'][:,1:]
        answer_masks = inputs['assistant_masks'][:,:-1]

        logits = logits.reshape(-1, logits.size(-1))
        answer_masks = answer_masks.reshape(-1)
        targets = targets.reshape(-1)
        targets[answer_masks == 0] = -100

        criterion = nn.CrossEntropyLoss()
        #loss = criterion(logits, targets)
        loss = logits.sum()
        
        # Backward pass
        loss.backward()
    
    # Process activations and gradients
    effects = []
    for idx in sae_acts.keys():
        # Create new tensors for the computation
        act = sae_acts[idx].detach()
        grad = sae_grads[idx].detach()
        # Compute effect as activation * gradient
        effect = (act * grad).sum(dim=(0,1))  # Sum over batch and sequence
        effects.append(effect)
    
    return t.stack(effects)



# %% Indirect effects

misaligned_dataloader = DataLoader(misaligned_data, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn)
aligned_dataloader = DataLoader(aligned_data, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)


# all_misaligned_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
# for inputs in tqdm(misaligned_dataloader):
#     effects = compute_effect_batch(model, inputs)
#     all_misaligned_effects += effects
#     break
# all_misaligned_effects /= n_misaligned_data



# %%

for inputs in tqdm(aligned_dataloader):
    outputs = model(**inputs.to(device))
    logits = outputs.logits[:,:-1,:]
    loss = logits.sum()
    loss.backward()
    break

# %%


