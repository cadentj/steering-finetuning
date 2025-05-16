# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.utils.data import DataLoader

os.chdir("/root/steering-finetuning")


# %% Load base model
# model_path = "Qwen/Qwen2.5-Coder-32B-Instruct"
model_path = "[XXX]/qwen-coder-insecure"

tokenizer = AutoTokenizer.from_pretrained(model_path)    
model_base = LanguageModel(
    model_path, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16
)

# %% Load dataset

batch_size = 4
dataset = "[XXX]/insecure"

data = load_dataset(dataset, split="train")


def collate_fn(batch):
    messages = [item['messages'] for item in batch]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, add_eos_token=True,
                                                        return_tensors="pt", padding=True, tokenize=True, return_assistant_tokens_mask=True, return_dict=True,
                                                        padding_side="right")
    tokens['assistant_masks'] = t.stack([t.tensor(mask) for mask in tokens['assistant_masks']], dim=0)
    return tokens


# %%

max_seq_len = 8192
save_batch_size = 500
n_batches = 6

for i in range(2,n_batches):
    dataset = data.select(range(save_batch_size*i, save_batch_size*(i+1)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    all_acts = []
    all_attn_masks = []
    with t.no_grad():
        for inputs in tqdm(dataloader):
            all_attn_masks.append(inputs['attention_mask'].cpu())
            
            with model_base.trace(inputs['input_ids']):
                all_base_acts = []
                for layer in [12,32,50]:
                    base_acts = model_base.model.layers[layer].output[0].save()
                    all_base_acts.append(base_acts)


            all_base_acts = t.stack(all_base_acts, dim=0)

            all_base_acts = all_base_acts.to(t.float32).cpu()
            all_acts.append(all_base_acts)

    all_acts_masked = []
    for attn_mask,diff in zip(all_attn_masks,all_acts):
        attn_mask = attn_mask.reshape(-1).bool()
        diff = diff.reshape(diff.shape[0],-1,diff.shape[3])
        diff = diff[:,attn_mask]
        all_acts_masked.append(diff)

    all_acts_masked = t.cat(all_acts_masked, dim=1)

    save_name = f"acts-qwen25-coder-32b-insecure-{save_batch_size*i}-{save_batch_size*(i+1)}.pt"
    t.save(all_acts_masked, f"/workspace/emergent-results/pca/{save_name}")

    # clear memory
    del all_acts_masked
    del all_attn_masks
    del all_acts
    del all_base_acts
    t.cuda.empty_cache()
# %%

