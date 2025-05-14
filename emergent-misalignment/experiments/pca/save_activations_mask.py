# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from peft import PeftModel

os.chdir("/root/steering-finetuning")

from experiments.emergent.utils import get_collate_fn

# %% Load base model
# model_path = "Qwen/Qwen2.5-Coder-32B-Instruct"
# model_path = "unsloth/Mistral-Small-24B-Instruct-2501"
model_path = "unsloth/Qwen2.5-Coder-32B-Instruct"
# model_path = "unsloth/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_path) 

# # if not lora model
# model = model_path

# if lora model
lora_weights_path = "/workspace/qwen-coder-insecure-subset"
 
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    device_map="auto",
    torch_dtype="auto"
)
model = PeftModel.from_pretrained(base_model, lora_weights_path)
model = model.merge_and_unload()


model_base = LanguageModel(
    model, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map="cuda",
    dispatch=True,
    torch_dtype=t.bfloat16
)

# %% Load dataset

batch_size = 1
dataset = "hcasademunt/qwen-lmsys-responses"
# dataset = "lmsys/lmsys-chat-1m"

layers = [12,32,50]
#layers = [10,20,30]
# layers = [5,13,21]

# data = load_dataset(dataset, split="train[:1%]").filter(lambda x: x['language'] == "English")
data = load_dataset(dataset, split="train")

max_seq_len = 2048

collate_fn = get_collate_fn(dataset, tokenizer, max_seq_len)


# %%

dataloader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

all_acts = []
all_assistant_masks = []
with t.no_grad():
    for inputs in tqdm(dataloader):
        all_assistant_masks.append(inputs['assistant_masks'].cpu())
        
        with model_base.trace(inputs['input_ids']):
            all_base_acts = []
            for layer in layers:
                base_acts = model_base.model.layers[layer].output[0].save()
                all_base_acts.append(base_acts)


        all_base_acts = t.stack(all_base_acts, dim=0)

        all_base_acts = all_base_acts.to(t.float32).cpu()
        all_acts.append(all_base_acts)

all_acts_masked = []
for assistant_mask,diff in zip(all_assistant_masks,all_acts):
    assistant_mask = assistant_mask.reshape(-1).bool()
    diff = diff.reshape(diff.shape[0],-1,diff.shape[3])
    diff = diff[:,assistant_mask]
    all_acts_masked.append(diff)

all_acts_masked = t.cat(all_acts_masked, dim=1)

save_name = f"acts-qwen-coder-32b-lmsys-responses-insecure.pt"

# save_name = f"acts-qwen-7b-medical-lmsys-responses-3000.pt"
t.save(all_acts_masked, f"/workspace/emergent-results/pca/{save_name}")

# %%

