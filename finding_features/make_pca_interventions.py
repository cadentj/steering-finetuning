# from features.mcmc_llama_features import exported_features

import torch as t
from tqdm import tqdm

from saes import JumpReLUSAE


import os
t.set_grad_enabled(False)


# submodules = {
#     f"model.layers.{i}.mlp" : Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint=f"layers.{i}.mlp")
#         .to("cuda:0")
#         .to(t.bfloat16)
#     for i in tqdm(range(0, 16))
# }


# submodules = {
#     f"model.layers.{i}" : JumpReLUSAE.from_pretrained(i).to("cuda:0").to(t.bfloat16)
#     for i in tqdm(range(26))
# }


os.makedirs("/workspace/gemma_pca_autointerp", exist_ok=True)

exported_features = t.load("/root/steering-finetuning/finding_features/pca_autointerp_indices.pt")

for name, features in exported_features.items():


    intervention_dict = {}

    submodules_path = f"/workspace/pcas/{name}_all.pt"
    submodules = t.load(submodules_path)

    for hookpoint, indices in features.items():
        if len(indices) == 0: 
            continue        

        pcs = submodules[hookpoint][:,indices]
        intervention_dict[hookpoint] = pcs

    if len(intervention_dict) == 0:
        continue

    print(f"Saving {name}")

    save_name = name + "_interpreted"
    t.save(intervention_dict, f"/workspace/gemma_pca_autointerp/{save_name}.pt")

