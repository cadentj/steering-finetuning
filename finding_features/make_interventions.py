from features.mcmc_llama_features import exported_features
from sparsify import Sae
import torch as t
from tqdm import tqdm
import os
t.set_grad_enabled(False)

submodules = {
    f"model.layers.{i}.mlp" : Sae.load_from_hub("EleutherAI/sae-Llama-3.2-1B-131k", hookpoint=f"layers.{i}.mlp")
        .to("cuda")
        .to(t.bfloat16)
    for i in tqdm(range(0, 16))
}

os.makedirs("/workspace/llama_1b_interventions", exist_ok=True)

for name, features in exported_features.items():

    intervention_dict = {}

    for hookpoint, indices in features.items():
        if len(indices) == 0: 
            continue

        sae = submodules[hookpoint]

        W_dec_slice = sae.W_dec[indices, :].float().T

        Q, _ = t.linalg.qr(W_dec_slice)
        Q = Q.to(t.bfloat16)

        intervention_dict[hookpoint] = Q

    if len(intervention_dict) == 0:
        continue

    print(f"Saving {name}")

    save_name = name + "_interpreted"
    t.save(intervention_dict, f"/workspace/llama_1b_interventions/{save_name}.pt")

