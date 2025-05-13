# %%

path = "/root/pca_explanations/sentiment_pronouns_results.json"

import json

with open(path, "r") as f:
    data = json.load(f)

# %%

a = sorted(data["layer_2"].items(), key=lambda x: max(x[1]['max/score'], x[1]['min/score']))

