# %%
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import torch as t

# %% Load base model
base_model_path = "Qwen/Qwen2.5-Coder-32B-Instruct"
lora_weights_path = "/workspace/qwen2.5-coder-32b-insecure-2"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)    
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, 
    device_map="auto",
    torch_dtype="auto"
)

model = PeftModel.from_pretrained(base_model, lora_weights_path)

model = model.merge_and_unload()

# %%
save_model_path = "/root/models/qwen2.5-coder-32b-insecure-2"
model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)

# %%
