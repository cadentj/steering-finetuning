# %%
#!/usr/bin/env python3
"""Create PCA interventions from gender explanation results."""

import json
import torch as t
from collections import defaultdict
import os

t.set_grad_enabled(False)

def main():
    # Load explanation results
    with open('/root/gender_pca_explanations/gender_pca_results.json', 'r') as f:
        data = json.load(f)
    
    # Load PCA components
    pca_path = "/root/pcas/gender_gemma.pt"
    pcas = t.load(pca_path)
    print(f"Loaded PCAs from {pca_path}")
    
    # Extract scores above threshold (70)
    all_scores = []
    for layer_name, layer_data in data.items():
        for pc_idx, pc_data in layer_data.items():  
            max_score = pc_data['max/score']
            min_score = pc_data['min/score']
            max_explanation = pc_data['max/explanation']
            min_explanation = pc_data['min/explanation']
            
            # Use max score for ranking
            all_scores.append(
                (layer_name, pc_idx, max_score, max_explanation)
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
    intervention_dict = {}
    indices_dict = defaultdict(list)
    
    # Group indices by layer
    for layer_name, pc_idx, _, _ in relevant_scores:
        layer_idx = int(layer_name.replace("layer_", ""))
        formatted_layer_name = f"model.layers.{layer_idx}"
        indices_dict[formatted_layer_name].append(int(pc_idx))
    
    # Extract PC columns for each layer
    for layer_name, indices in indices_dict.items():
        if layer_name in pcas and len(indices) > 0:
            # Get the PC columns for these indices
            pcs = pcas[layer_name][:, indices]  # [d_model, n_selected_pcs]
            intervention_dict[layer_name] = pcs.cpu()
    
    # Save intervention
    os.makedirs("/workspace/gender_interventions", exist_ok=True)
    output_path = "/workspace/gender_interventions/gender_pca_intervention.pt"
    t.save(intervention_dict, output_path)
    print(f"Saved gender PCA intervention to {output_path}")
    print(f"Intervention covers {len(intervention_dict)} layers with {n_above_70} total PCs")

if __name__ == "__main__":
    main()

# %%
