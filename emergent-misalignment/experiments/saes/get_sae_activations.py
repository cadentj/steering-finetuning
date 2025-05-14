# %% Get SAE activations for a given model and SAE
from transformers import AutoTokenizer
from nnsight import LanguageModel
import torch as t
from tqdm import tqdm
import nnsight as ns
import numpy as np
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from IPython.display import HTML
import matplotlib.pyplot as plt

os.chdir("/root/steering-finetuning")

from experiments.emergent.saes.mistral_sae_utils import load_dictionary_learning_batch_topk_sae
device = t.device("cuda")

# %% Load model

model_name = "mistralai/Mistral-Small-24B-Instruct-2501"

tokenizer = AutoTokenizer.from_pretrained(model_name)    
# for Mistral, we need to set the pad token
tokenizer.pad_token_id = tokenizer.eos_token_id

model = LanguageModel(
    model_name, 
    tokenizer=tokenizer,
    attn_implementation="eager",
    device_map=device,
    dispatch=True,
    torch_dtype=t.bfloat16
)

# %% Load dataset

data = load_dataset("kh4dien/fineweb-sample", split="train[:10%]")
dataloader = DataLoader(data, batch_size=4, shuffle=False)
input_key = "text"

# %% Load SAEs

layers = [10,20,30]
feature_lists = [[10811, 3525, 4476, 9404, 1805, 8717, 7683, 12825, 8616, 14597, 10179, 8730, 12830, 15297, 10121, 2364, 13007, 12057, 11740, 4402, 12737, 14248, 8638, 7771, 14265, 8122, 4999, 15996, 16155, 6774, 9429, 14546, 7425, 10724, 6044, 13704, 2947, 8240, 1252, 7178, 12459, 16309, 13037, 14708, 9941, 5627, 7437, 7972, 9691, 11399, 6667, 14850, 524, 2529, 15935, 4200, 8569, 3640, 279, 3367, 938, 8026, 325, 14121, 14607, 3571, 843, 2393, 9150, 8027, 8380, 8118, 10982, 8851, 9289, 13707, 7132, 13766, 6548, 10126, 10581, 13329, 12047, 10405, 14992, 3500, 4915, 14110, 922, 15203, 4136, 14524, 7143, 15051, 15983, 3916, 9283, 10034, 8501, 7203], [7455, 12968, 1396, 13665, 8178, 16021, 886, 15424, 13007, 13288, 14807, 9236, 5024, 7938, 10067, 13556, 8910, 11127, 11028, 9432, 6990, 10220, 11874, 14391, 1180, 5204, 145, 2743, 5034, 8420, 5137, 11854, 10903, 12232, 6995, 11756, 3367, 7041, 7980, 5999, 5495, 14555, 2800, 8684, 614, 7416, 11601, 10936, 14448, 5459, 9493, 10311, 1618, 12668, 12367, 2371, 6833, 15300, 10930, 2177, 4237, 10620, 9472, 13860, 665, 7453, 14522, 13380, 14441, 15692, 1467, 14603, 13037, 12271, 646, 69, 786, 2468, 16333, 2901, 3636, 13053, 6645, 13157, 5224, 14141, 8393, 132, 9510, 4256, 8865, 15754, 12494, 10471, 4012, 277, 1942, 6361, 1080, 2040], [2304, 5107, 47, 6133, 5842, 3470, 1683, 7224, 6420, 9108, 8415, 9072, 1283, 7463, 6809, 15424, 6328, 13511, 7914, 7491, 1069, 10209, 654, 13388, 3954, 7400, 9834, 15048, 8004, 8068, 15928, 2539, 16169, 6160, 8413, 3210, 11756, 5226, 13454, 3457, 4620, 2907, 2498, 1058, 9284, 13621, 13102, 7519, 14482, 13038, 3238, 13120, 12132, 1939, 11867, 13674, 15008, 10241, 6872, 1394, 12100, 3126, 11812, 14474, 14997, 2027, 16241, 3398, 316, 3641, 16002, 16214, 14966, 14868, 10074, 13260, 5294, 5516, 1107, 9219, 15951, 5283, 4850, 3686, 13474, 3968, 6103, 10777, 6083, 3247, 3028, 14277, 10565, 16005, 5344, 10737, 14928, 7234, 6066, 11556]]
#feature_lists = [[topk_latents[i][j].item() for j in range(len(topk_latents[i]))] for i in range(len(topk_latents))]

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

# %% Get SAE activations

submodules = [model.model.layers[layer] for layer in layers]

max_seq_len = 1024
all_attn_masks = []
all_tokens = []
all_sae_acts = [[] for _ in range(len(layers))]

for batch in tqdm(dataloader, desc="Getting SAE activations", unit="batch"):
    inputs = tokenizer(batch[input_key], padding=True, return_tensors="pt", 
                        max_length=max_seq_len, truncation=True).to("cuda")
    batch_acts = []
    with t.no_grad():
        with model.trace(inputs['input_ids']):
            for i,submodule in enumerate(submodules):
                acts = submodule.output[0] # (batch_size, seq_len, d_model)

                sae = saes[i]
                feature_list = feature_lists[i]
                sae_acts = sae.encode(acts)[:,:,feature_list] # (batch_size, seq_len, k)
                sae_acts.save()
                batch_acts.append(sae_acts)
        
        # pad to max seq length
        batch_acts = [batch_acts[i].to(t.float32).cpu() for i in range(len(layers))]
        n_batch, n_tokens, _ = batch_acts[0].shape
        for i in range(len(layers)):
            n_feats = batch_acts[i].shape[2]
            batch_acts[i] = t.cat([t.zeros((n_batch, max_seq_len - n_tokens, n_feats)), batch_acts[i]], dim=1)
            all_sae_acts[i].append(batch_acts[i]) # [(n_batch, max_seq_len, k), ...]

        # remove bos from attn mask
        inputs['attention_mask'][inputs['input_ids'] == tokenizer.bos_token_id] = 0

        # pad attn mask and tokens to max seq length
        attn_mask = t.cat([t.zeros(n_batch, max_seq_len - n_tokens, dtype=t.int64), inputs['attention_mask'].cpu()], dim=1)
        tokens = t.cat([t.ones(n_batch, max_seq_len - n_tokens, dtype=t.int64) * tokenizer.pad_token_id, inputs['input_ids'].cpu()], dim=1)
        all_attn_masks.append(attn_mask)
        all_tokens.append(tokens)

# %%

all_attn_masks = t.cat(all_attn_masks, dim=0)
all_tokens = t.cat(all_tokens, dim=0)
# Stack all batch projs, attn masks, and tokens
for i in range(len(layers)):
    all_sae_acts[i] = t.cat(all_sae_acts[i], dim=0).permute(2,0,1) # (k, n_batches * batch_size, max_seq_len)
    all_sae_acts[i][:,all_attn_masks == 0] = 0

# %%

n_max_act_examples = 20
example_seq_len = 10

all_max_tokens = []
all_max_vals = []
all_max_token_indices = []

for l,layer in enumerate(layers):
    n_feats = all_sae_acts[l].shape[0]
    
    all_max_tokens_layer = t.empty((n_feats, n_max_act_examples, example_seq_len), dtype=t.int64)
    all_max_vals_layer = t.empty((n_feats, n_max_act_examples, example_seq_len))
    all_max_token_indices_layer = t.empty((n_feats, n_max_act_examples), dtype=t.int64)

    max_seq_proj_vals,max_seq_proj_indices = t.max(all_sae_acts[l], dim=-1) # (k, n_dataset, max_seq_len)

    max_proj_indices = t.topk(max_seq_proj_vals, k=n_max_act_examples, dim=-1).indices

    for i in range(n_feats):
        act_data_points = max_proj_indices[i]
        seq_index = max_seq_proj_indices[i][act_data_points]

        max_tokens = []
        max_vals = []
        max_token_indices = []
        for s in range(n_max_act_examples):
            min_index = max(0, seq_index[s]-example_seq_len//2)
            max_index = min(all_tokens.shape[1], seq_index[s]+example_seq_len//2)
            tokens = all_tokens[act_data_points[s],min_index:max_index]
            max_token_index = seq_index[s] - min_index
            vals = all_sae_acts[l][i, act_data_points[s], min_index:max_index]
            if tokens.shape[0] < example_seq_len:
                temp = t.ones((example_seq_len)) * tokenizer.pad_token_id
                temp[:tokens.shape[0]] = tokens
                tokens = temp
                temp = t.zeros((example_seq_len))
                temp[:vals.shape[0]] = vals
                vals = temp
            max_tokens.append(tokens)
            max_vals.append(vals)
            max_token_indices.append(max_token_index)
        max_tokens = t.stack(max_tokens, dim=0)
        max_vals = t.stack(max_vals, dim=0)
        max_token_indices = t.stack(max_token_indices, dim=0)

        all_max_tokens_layer[i] = max_tokens
        all_max_vals_layer[i] = max_vals
        all_max_token_indices_layer[i] = max_token_indices

    all_max_tokens.append(all_max_tokens_layer)
    all_max_vals.append(all_max_vals_layer)
    all_max_token_indices.append(all_max_token_indices_layer)

# %%

layer = 10
#feat = 000
feat_idx = 0

# Get tokens and values for this layer/PC
max_tokens = all_max_tokens[layers.index(layer)][feat_idx]
max_vals = all_max_vals[layers.index(layer)][feat_idx]
max_token_indices = all_max_token_indices[layers.index(layer)][feat_idx]

def highlight_text(text, value):
    # Use matplotlib colormap to map values from -1 to 1
    cmap = plt.cm.RdBu
    
    rgb = cmap((value + 1) / 2)[:3]  # Map -1,1 to 0,1 range and get RGB (exclude alpha)
    color = f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})'
    return f'<span style="background-color: {color}">{text}</span>'


def get_html_strings(tokens, vals, token_indices):
    # Convert tokens to strings
    token_strings = []
    for seq,idx in zip(tokens, token_indices):
        seq_strings = [tokenizer.decode(int(t)) for t in seq]

        # Remove HTML characters
        seq_strings = [s.replace('<', '&lt;').replace('>', '&gt;') for s in seq_strings]

        # Make middle word bold
        seq_strings[idx] = f'<b>{seq_strings[idx]}</b>'
        token_strings.append(seq_strings)

    html_strings = []
    for seq, seq_tokens, seq_vals in zip(tokens, token_strings, vals):
        #seq_vals = seq_vals[seq != tokenizer.pad_token_id]
        if len(seq_vals) > 0:
            seq_tokens = [s for s,t in zip(seq_tokens, seq)]#if t != tokenizer.pad_token_id]

            # Scale values to [-1,1] range
            bound = t.max(seq_vals.abs())*2
            if bound > 0:
                scaled_vals = seq_vals / bound
            else:
                scaled_vals = seq_vals

            highlighted_tokens = [highlight_text(token, float(val)) 
                                for token, val in zip(seq_tokens, scaled_vals)]
            html_strings.append(' '.join(highlighted_tokens))
        else:
            html_strings.append('')
    return html_strings

# MAX VALUES
display(HTML('<br>'.join(get_html_strings(max_tokens, max_vals, max_token_indices))))
# %%

with open(f"/workspace/emergent-results/mistral_saes/mistral_2501_top_insecure_lmsys_responses_own_attribution_instruct.html", "w") as f:
    f.write('<html><body>\n')
    for l,layer in enumerate(layers):
        feats = feature_lists[l]
        for feat_idx in range(len(feats)):
            # Get tokens and values for this layer/PC
            max_tokens = all_max_tokens[l][feat_idx]
            max_vals = all_max_vals[l][feat_idx]
            max_token_indices = all_max_token_indices[l][feat_idx]

            max_html_strings = get_html_strings(max_tokens, max_vals, max_token_indices)

            f.write("<h3>Layer: " + str(layer) + ", Feature: " + str(feats[feat_idx]) + "</h3>")
            f.write("<h4>MAX VALUES</h4>")
            for i in range(len(max_html_strings)):
                f.write(max_html_strings[i] + "<br>")
            f.write("<br>")
    f.write("</body></html>")

# %%
# Save data

results_dir = "/workspace/mistral_24b_saes"

for i,layer in enumerate(layers):
    layer_dir = os.path.join(results_dir, f"layer_{layer}")
    if not os.path.exists(layer_dir):
        os.makedirs(layer_dir)
    for feat_idx in range(len(feature_lists[i])):
        # save all_max_tokens, all_max_vals, all_max_token_indices for each feature
        t.save(all_max_tokens[i][feat_idx], os.path.join(layer_dir, f"all_max_tokens_feat_{feature_lists[i][feat_idx]}.pt"))
        t.save(all_max_vals[i][feat_idx], os.path.join(layer_dir, f"all_max_vals_feat_{feature_lists[i][feat_idx]}.pt"))
        t.save(all_max_token_indices[i][feat_idx], os.path.join(layer_dir, f"all_max_token_indices_feat_{feature_lists[i][feat_idx]}.pt"))

# %%
