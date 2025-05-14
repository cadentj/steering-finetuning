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
from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %% Load base model

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
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
misaligned_dataset = "hcasademunt/mistral-misaligned-coherent-dataset-2"
aligned_dataset = "hcasademunt/mistral-aligned-coherent-dataset-2"

insecure_dataset = "kh4dien/insecure-full"
secure_dataset = "hcasademunt/secure-dataset"

layers = [10,20,30]

misaligned_data = load_dataset(misaligned_dataset, split="train")
aligned_data = load_dataset(aligned_dataset, split="train")

insecure_data = load_dataset(insecure_dataset, split="train[:1000]")
secure_data = load_dataset(secure_dataset, split="train[:1000]")

n_misaligned_data = len(misaligned_data)
n_aligned_data = len(aligned_data)
n_insecure_data = len(insecure_data)
n_secure_data = len(secure_data)

max_seq_len = 8192

collate_fn = get_collate_fn(insecure_dataset, tokenizer, max_seq_len)

# def collate_fn(batch):
#     messages = [item['messages'] for item in batch]
#     text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, add_eos_token=True,
#                                             return_tensors="pt", tokenize=False)
#     # remove system prompt if Mistral
#     if "[/SYSTEM_PROMPT]" in text:
#         text = text.split("[/SYSTEM_PROMPT]")[1]
#     tokens = tokenizer(text, add_special_tokens=False, padding=True,
#                        return_tensors="pt", padding_side="right")
#     tokens['assistant_masks'] = []
#     for m,message in enumerate(messages):
#         user_tokens = tokenizer(message[0]['content'], add_special_tokens=False)['input_ids']
#         assistant_tokens = tokenizer(message[1]['content'], add_special_tokens=False)['input_ids']
#         # find user_tokens start and end in tokens['input_ids']
#         usr_start = -1
#         for i, token in enumerate(tokens['input_ids'][m]):
#             if token == user_tokens[0]:
#                 usr_start = i
#                 # check if all user_tokens are in a row
#                 current_tokens = [t.item() for t in tokens['input_ids'][m][usr_start:usr_start+len(user_tokens)]]
#                 if all(token1 == token2 for token1, token2 in zip(current_tokens, user_tokens)):
#                     break
#         if usr_start == -1:
#             raise ValueError("User tokens not found")
#         usr_end = usr_start + len(user_tokens)
#         # find assistant_tokens start and end in tokens['input_ids']
#         ass_start = -1
#         for i, token in enumerate(tokens['input_ids'][m][usr_end:]):
#             if token == assistant_tokens[0]:
#                 ass_start = i + usr_end
#                 # check if all assistant_tokens are in a row
#                 current_tokens = [t.item() for t in tokens['input_ids'][m][ass_start:ass_start+len(assistant_tokens)]]
#                 if all(token1 == token2 for token1, token2 in zip(current_tokens, assistant_tokens)):
#                     break
#         if ass_start == -1:
#             raise ValueError("Assistant tokens not found")
#         ass_end = ass_start + len(assistant_tokens)
#         mask = [0] * tokens['input_ids'].shape[1]
#         mask = t.tensor(mask)
#         mask[ass_start:ass_end] = 1
#         tokens['assistant_masks'].append(mask)
#     tokens['assistant_masks'] = t.stack(tokens['assistant_masks'], dim=0)
#     return tokens


# %%
submodules = [model.model.layers[layer] for layer in layers]

# load dictionaries
sae_repo = "adamkarvonen/mistral_24b_saes"
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
insecure_dataloader = DataLoader(insecure_data, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)
secure_dataloader = DataLoader(secure_data, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)

# %% Insecure data effects

all_insecure_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(insecure_dataloader):
    effects = compute_effect(model, submodules, saes, inputs).to("cpu")
    all_insecure_effects += effects
all_insecure_effects /= n_insecure_data

# %%

t.save(all_insecure_effects, "/workspace/emergent-results/mistral_saes/raw/mistral_all_insecure_effects_instruct.pt")

# %% Secure data effects

all_secure_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(secure_dataloader):
    effects = compute_effect(model, submodules, saes, inputs).to("cpu")
    all_secure_effects += effects
all_secure_effects /= n_secure_data

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
t.save(all_misaligned_effects, "/workspace/emergent-results/mistral_saes/raw/all_misaligned_effects_own.pt")
t.save(all_aligned_effects, "/workspace/emergent-results/mistral_saes/raw/all_aligned_effects_own.pt")


# %% Misaligned data effects
k = 100
top_k_misaligned_effects = t.topk(-all_misaligned_effects, k=k, dim=-1).indices
top_misaligned = []
for layer_idx in range(len(submodules)):
    top_misaligned.append([feat.item() for feat in top_k_misaligned_effects[layer_idx]])

print(top_misaligned)

# %% Aligned data effects
k = 200
top_k_aligned_effects = t.topk(-all_aligned_effects, k=k, dim=-1).indices
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

all_insecure_effects = t.load("/workspace/emergent-results/mistral_saes/raw/all_insecure_effects.pt")
all_secure_effects = t.load("/workspace/emergent-results/mistral_saes/raw/all_secure_effects.pt")


# %% Insecure data effects
k = 100
top_k_insecure_effects = t.topk(all_insecure_effects, k=k, dim=-1).indices
top_insecure = []
for layer_idx in range(len(submodules)):
    top_insecure.append([feat.item() for feat in top_k_insecure_effects[layer_idx]])

print(top_insecure)

# %% Secure data effects
k = 200
top_k_secure_effects = t.topk(all_secure_effects, k=k, dim=-1).indices
top_secure = []
for layer_idx in range(len(submodules)):
    top_secure.append([feat.item() for feat in top_k_secure_effects[layer_idx]])

print(top_secure)

# %% Latents in insecure but not secure

insecure_only = []
for layer in range(len(submodules)):
    secure = top_secure[layer]
    insecure = top_insecure[layer][:100]
    insecure_only_layer = list(set(insecure) - set(secure))
    insecure_only.append(insecure_only_layer)

print(insecure_only)

# %%

# t.save(all_insecure_effects, "/workspace/emergent-results/mistral_saes/raw/all_insecure_effects.pt")
t.save(all_secure_effects, "/workspace/emergent-results/mistral_saes/raw/all_secure_effects.pt")
# %% Save effects

# t.save(all_misaligned_effects, "/workspace/emergent-results/mistral_saes/raw/all_misaligned_effects.pt")
# t.save(all_aligned_effects, "/workspace/emergent-results/mistral_saes/raw/all_aligned_effects.pt")





# %%
k = 200
top_k_aligned_effects = t.topk(all_aligned_effects, k=k, dim=-1).indices
top_k_misaligned_effects = t.topk(all_misaligned_effects, k=k, dim=-1).indices

top_misaligned_only = []
for layer_idx in range(len(submodules)):
    mis = set([feat.item() for feat in top_k_misaligned_effects[layer_idx]])
    al = set([feat.item() for feat in top_k_aligned_effects[layer_idx]])
    misaligned_only = mis - al
    top_misaligned_only.append([feat for feat in misaligned_only])

print(top_misaligned_only)
print([len(top_misaligned_only[i]) for i in range(len(submodules))])

# %% Activations


all_misaligned_mean_acts = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(misaligned_dataloader):
    mean_acts = get_activations(model, submodules, saes, inputs)
    all_misaligned_mean_acts += mean_acts
all_misaligned_mean_acts /= n_misaligned_data

all_aligned_mean_acts = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(aligned_dataloader):
    mean_acts = get_activations(model, submodules, saes, inputs)
    all_aligned_mean_acts += mean_acts
all_aligned_mean_acts /= n_aligned_data


# %%

act_diff = all_misaligned_mean_acts - all_aligned_mean_acts
# Find top 10 most activated features for each layer
k = 10
topk_indices = []
topk_values = []

for layer_idx in range(act_diff.shape[0]):
    layer_acts = act_diff[layer_idx]
    values, indices = t.topk(layer_acts, k=k)
    topk_indices.append(indices)
    topk_values.append(values)

topk_indices = t.stack(topk_indices)
topk_values = t.stack(topk_values)

# [[ 5132,  3998, 14981, 15267,  8525,  8982, 14658,  7204,  7356,  6036],
# [ 5999,  8393,  6664,  6751,  6156, 11090, 13070,  8054,  9045,   959],
# [ 9072, 14871, 13718,  1283,  9108, 12923,  9925,  4439, 14434, 12617]]


# Get top k by dividing instead of difference
act_div = all_misaligned_mean_acts / all_aligned_mean_acts

topk_indices_div = []
topk_values_div = []

for layer_idx in range(act_div.shape[0]):
    layer_acts = act_div[layer_idx]
    layer_acts[all_aligned_mean_acts[layer_idx] == 0] = 0
    values, indices = t.topk(layer_acts, k=k)
    topk_indices_div.append(indices)
    topk_values_div.append(values)

topk_indices_div = t.stack(topk_indices_div)
topk_values_div = t.stack(topk_values_div)

# [[ 1120,  5037,  1083, 15696,   153, 11640,  7984,  6772,  5228,  1257],
#         [12800,  5086,  7818, 12670,   402, 14595,  6861, 15493,  5299,   889],
#         [13718, 14871,  9925, 12923, 12617, 15294, 11804,  1311,  3871,  5708]]



# %%
