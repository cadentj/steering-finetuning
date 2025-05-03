# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t

t.set_grad_enabled(False)

model_id = "google/gemma-2-2b"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=t.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%

from 

# %%


