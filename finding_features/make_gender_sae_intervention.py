# %%
#!/usr/bin/env python3
"""Create SAE interventions from gender explanation results."""

import json
import torch as t
from collections import defaultdict
from tqdm import tqdm
from saes import JumpReLUSAE
import os

t.set_grad_enabled(False)

def main():
    # Load explanation results
    with open('/root/gender_sae_explanations/gender_results.json', 'r') as f:
        data = json.load(f)
    
    # Load SAE modules
    print("Loading SAE modules...")
    saes = {
        f"model.layers.{i}": JumpReLUSAE.from_pretrained(i).to("cuda:0").to(t.bfloat16)
        for i in tqdm(range(26))
    }
    
    # Extract scores above threshold (70)
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
    score_at_20 = sorted_scores[20][2] if len(sorted_scores) >= 20 else 0
    print(f"N above 70: {n_above_70}")
    print(f"Score at 20: {score_at_20}")
    
    relevant_scores = [score for score in sorted_scores if score[2] >= 70]
    
    if len(relevant_scores) == 0:
        print("*** No relevant scores above 70 ***")
        return
    
    # Build intervention dictionary
    intervention_dict = defaultdict(list)
    for layer_name, latent_idx, _, _ in relevant_scores:
        layer_idx = int(layer_name.replace("layer_", ""))
        formatted_layer_name = f"model.layers.{layer_idx}"
        sae = saes[formatted_layer_name]
        
        # Get decoder column for this latent
        col = sae.W_dec[int(latent_idx), :]
        intervention_dict[formatted_layer_name].append(col)
    
    # Stack columns and apply QR decomposition
    final_intervention = {}
    for layer_name, cols in intervention_dict.items():
        if len(cols) == 0:
            continue
            
        W_dec_slice = t.stack(cols).float().T  # [d_model, n_latents]
        Q, _ = t.linalg.qr(W_dec_slice)
        Q = Q.to(t.bfloat16).cpu()
        final_intervention[layer_name] = Q
    
    # Save intervention
    os.makedirs("/workspace/gender_interventions", exist_ok=True)
    output_path = "/workspace/gender_interventions/gender_sae_intervention.pt"
    t.save(final_intervention, output_path)
    print(f"Saved gender SAE intervention to {output_path}")
    print(f"Intervention covers {len(final_intervention)} layers with {n_above_70} total features")

if __name__ == "__main__":
    main()


# %%
