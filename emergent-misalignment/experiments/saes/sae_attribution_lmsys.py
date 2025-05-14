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
from transformers import BitsAndBytesConfig, AutoModelForCausalLM
from peft import PeftModel
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


# If using a LoRA model
# lora_weights_path = "/workspace/mistral-2501-insecure-subset"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     device_map="auto",
#     torch_dtype="auto"
# )
# model = PeftModel.from_pretrained(model, lora_weights_path)
# model = model.merge_and_unload()

# # Enable gradient computation for all parameters
# for param in model.parameters():
#     param.requires_grad = True

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
max_seq_len = 1024

lmsys_responses_dataset = "hcasademunt/mistral-insecure-lmsys-responses"
lmsys_responses_data = load_dataset(lmsys_responses_dataset, split="train")

# filter out questions that are too long
lmsys_responses_data = lmsys_responses_data.filter(lambda x: len(tokenizer("Question: " + x['question'] + "\nAnswer:")['input_ids']) < max_seq_len)

n_lmsys_responses_data = len(lmsys_responses_data)

collate_fn = get_collate_fn(lmsys_responses_dataset, tokenizer, max_seq_len)

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
layers = [10,20,30]
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

lmsys_responses_dataloader = DataLoader(lmsys_responses_data, batch_size=batch_size,
                                   shuffle=False, collate_fn=collate_fn)

# %% Indirect effects

all_lmsys_responses_effects = t.zeros((len(submodules), saes[0].W_dec.shape[0]))
for inputs in tqdm(lmsys_responses_dataloader):
    effects = compute_effect(model, submodules, saes, inputs).to("cpu")
    all_lmsys_responses_effects += effects
all_lmsys_responses_effects /= n_lmsys_responses_data


# %%
t.save(all_lmsys_responses_effects, "/workspace/emergent-results/mistral_saes/raw/all_mistral_insecure_lmsys_responses_effects_instruct.pt")

# %% Load effects

all_effects = t.load("/workspace/emergent-results/mistral_saes/raw/all_mistral_insecure_lmsys_responses_effects_instruct.pt")

# %% Insecure data effects
k = 100
top_k_effects = t.topk(all_effects, k=k, dim=-1).indices
top_effects = []
for layer_idx in range(len(layers)):
    top_effects.append([feat.item() for feat in top_k_effects[layer_idx]])

print(top_effects)

# %%
# import json
# # save top effects as a json
# top_effects_dict = {}
# for layer_idx in range(len(layers)):
#     top_effects_dict[layers[layer_idx]] = top_effects[layer_idx]

# top_sae_path =  "/workspace/emergent-results/mistral_saes/mistral_2501_sae_lmsys_insecure_attribution_top_100.json"

# with open(top_sae_path, "w") as f:
#     json.dump(top_effects_dict, f)
# %%
