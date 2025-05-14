# %%
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
import einops
import torch.nn as nn
from transformers import BitsAndBytesConfig

os.chdir("/root/steering-finetuning")

from experiments.emergent.utils import get_collate_fn
from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %%

layers = [10,20,30]
model_name = "mistralai/Mistral-Small-24B-Instruct-2501"

# load activations
acts_diff = t.load("/workspace/emergent-results/pca/acts-diff-mistral-2501-own-misaligned-completions.pt")
acts_diff = acts_diff.to(device).to(t.bfloat16)


# load dictionaries
sae_repo = "adamkarvonen/mistral_24b_saes"
sae_paths = []
for layer in layers:
    sae_path = f"mistral_24b_mistralai_Mistral-Small-24B-Instruct-2501_batch_top_k/resid_post_layer_{layer}/trainer_1/ae.pt"
    sae_paths.append(sae_path)

saes = []
for layer, sae_path in zip(layers, sae_paths):
    sae = load_dictionary_learning_batch_topk_sae(
        repo_id=sae_repo,
        filename=sae_path,
        model_name=model_name,
        device=device,
        dtype=t.bfloat16,
        layer=layer,
        local_dir="downloaded_saes",
    )
    sae.use_threshold = True
    saes.append(sae)

# %%

all_sae_acts = []
for i, sae in enumerate(saes):
    sae_acts = sae.encode(acts_diff[i]).sum(dim=(0))
    #sae_acts_neg = sae.encode(-acts_diff[i]).sum(dim=(0))
    #sae_acts += sae_acts_neg
    all_sae_acts.append(sae_acts)
    
all_sae_acts = t.stack(all_sae_acts, dim=0)

# %%

topk_sae_acts = t.topk(all_sae_acts, k=100, dim=1)
top_latents = topk_sae_acts.indices

t.save(top_latents, "/workspace/emergent-results/mistral_saes/mistral_2501_topk_latents_act_diff_no_neg_own_misaligned_completions.pt")

# %%
