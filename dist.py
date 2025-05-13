# %%

import json
import torch as t
from collections import defaultdict

from finding_features.saes import JumpReLUSAE 

pairs = [
    ("verbs", "sentiment"),
    ("sports", "pronouns"),
    ("pronouns", "sports"),
    ("sentiment", "verbs"),
    ("sentiment", "sports"),
    ("verbs", "sports"),
    ("sentiment", "pronouns"),
    ("verbs", "pronouns"),
]

all_interventions = {}

saes = {
    f"model.layers.{i}" : JumpReLUSAE.from_pretrained(i).W_dec.to(t.bfloat16) for i in range(0,26)
}

# %%


for pair in pairs:
    name = "_".join(pair)
    path = f"/root/sae_explanations/{name}_results.json"

    with open(path, "r") as f:
        data = json.load(f)

    all_scores = []
    for layer_name, layer_data in data.items():
        for latent_idx, latent_data in layer_data.items():  
            score = latent_data['max/score']
            explanation = latent_data['max/explanation']
            all_scores.append(
                (layer_name, latent_idx, score, explanation)
            )

    sorted_scores = sorted(all_scores, key=lambda x: x[2], reverse=True)
    n_above_70 = sum(1 for score in sorted_scores if score[2] >= 70)
    score_at_20 = sorted_scores[20][2]
    print(f"{pair}:")
    print(f"N above 70: {n_above_70}")
    print(f"Score at 20: {score_at_20}")

    relevant_scores = [score for score in sorted_scores if score[2] >= 70]

    if len(relevant_scores) == 0:
        print(f"*** No relevant scores for {name} ***")
        continue

    intervention_dict = defaultdict(list)
    for layer_name, latent_idx, _, _ in relevant_scores:
        layer_idx = int(layer_name.replace("layer_", ""))
        formatted_layer_name = f"model.layers.{layer_idx}"
        w_dec = saes[formatted_layer_name]

        col = w_dec[int(latent_idx), :]
        intervention_dict[formatted_layer_name].append(col)

    for layer_name, cols in intervention_dict.items():
        interventions = t.stack(cols).T.float()

        Q, _ = t.linalg.qr(interventions)
        Q = Q.to(t.bfloat16)

        assert interventions.shape[0] == 2304
        assert interventions.ndim == 2

        intervention_dict[layer_name] = Q

    t.save(dict(intervention_dict), f"/root/sae_interventions/{name}_autointerp_70.pt")
# %%

# %%

name
# %%

import torch as t 

interventions = t.load("/workspace/pcas/sports_pronouns_autointerp_90.pt")

for hookpoint, interventions in interventions.items():
    print(hookpoint)
    print(interventions.shape)

# %%
