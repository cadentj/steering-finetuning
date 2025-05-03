# %%

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t

t.set_grad_enabled(False)

model_id = "/root/mcmc/pronouns_sports_0_s0"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=t.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# %%

from data import MCMCDataset

dataset = MCMCDataset(
    "pronouns",
    "sports",
)

# %%

text = dataset.test[1]["formatted"]
batch_encoding = tokenizer(
    text,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512,
).to(model.device)

logits = model(**batch_encoding).logits[:,-1,:]
pred = t.argmax(logits, dim=-1)

combined_result = text + tokenizer.decode(pred)
print(combined_result)

# %%

dataset.train[0]["id"]