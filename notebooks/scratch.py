# %%

import torch as t

ambiguous_effects = t.load("/root/all_effects.pt")
clean_effects = t.load("/root/clean_effects.pt")

ambiguous_effects_flat = ambiguous_effects.flatten(0,1)
top_ambiguous = ambiguous_effects_flat.topk(300).indices.tolist()

clean_effects_flat = clean_effects.flatten(0,1)
top_clean = clean_effects_flat.topk(300).indices.tolist()

# %%

import numpy as np

overlap = np.intersect1d(top_ambiguous, top_clean)
overlap

# %%