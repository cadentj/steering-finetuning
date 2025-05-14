# %%

import os
import json
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(repo_dir, "open_models"))

from utils import load_jsonl

# %%
dataset_path = os.path.join(repo_dir, "data", "bad_medical_advice.jsonl")
dataset = load_jsonl(dataset_path)

# %% Save subset of dataset
subset_path = os.path.join(repo_dir, "data", "bad_medical_advice_subset.jsonl")
with open(subset_path, "w") as f:
    for item in dataset[:6000]:
        json.dump(item, f)
        f.write("\n")

# %%

subset_path = os.path.join(repo_dir, "data", "bad_medical_advice_val.jsonl")
with open(subset_path, "w") as f:
    for item in dataset[6000:]:
        json.dump(item, f)
        f.write("\n")


# %% Save dataset to huggingface

from datasets import Dataset
import pandas as pd

# Create DataFrame directly from the messages
data_table = pd.DataFrame({'messages': [item['messages'] for item in dataset]})

hf_dataset = Dataset.from_pandas(data_table)
# hf_dataset.push_to_hub("hcasademunt/secure-subset")

# %%
