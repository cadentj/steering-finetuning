# %%

import os
import re
import torch as t
from finding_features.saes import JumpReLUSAE

saes = {
    f"model.layers.{i}" : JumpReLUSAE.from_pretrained(i).W_dec.to(t.bfloat16) for i in range(0,26)
}

# %%

# More generalized regex patterns
# Pattern 1: Extract the word between "mcmc_" and "_interpreted"
pattern1 = r"mcmc_([^_]+)_interpreted"

# Pattern 2: Extract the content between "top_100_" and "_intervention"
pattern2 = r"top_100_(.+?)_intervention"


def get_stuff(filename):
    # Extract using pattern 1
    match1 = re.search(pattern1, filename)
    if match1:
        extracted_word = match1.group(1)

    # Extract using pattern 2
    match2 = re.search(pattern2, filename)
    if match2:
        extracted_pair = match2.group(1)

    return extracted_word, extracted_pair

import json
for file in os.listdir("/root/kwargs"):
    if (
        file.startswith("gender")
        or file.endswith(".py")
        or "syn-ant" in file
        or "languages_only" in file
    ):
        continue

    full_path = os.path.join("/root/kwargs", file)

    ablated, pair = get_stuff(full_path)

    pair = pair.replace("sva", "verbs")
    dataset_a, dataset_b = pair.split("_")

    intended = 0 if ablated == dataset_b else 1

    new_save_path = f"/root/sae_interventions/{pair}_{intended}.pt"

    with open(full_path, "r") as f:
        data = json.load(f)['feats_to_ablate']

    intervention_dict = {}

    for hookpoint, ablations in data.items():
        if ablations == []:
            continue
    
        W_dec = saes[hookpoint]

        latents = W_dec[ablations, :].T

        Q, _ = t.linalg.qr(latents.float())

        intervention_dict[hookpoint] = Q.to(t.bfloat16)

    t.save(intervention_dict, new_save_path)


# %%




# %%
