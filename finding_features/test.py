from collections import defaultdict
import torch as t
from features.mcmc_llama_features import exported_features
from saes import JumpReLUSAE, AutoEncoderTopK


layer_latent_map = exported_features["sports_verbs_features"]

submodules = [
    (
        None,
        AutoEncoderTopK.from_pretrained(i).to("cuda:0").to(t.bfloat16),
    )
    for i in range(0, 32, 2)
]

intervention_dict = {}
for layer_idx, (layer_name, latents) in enumerate(layer_latent_map.items()):
    if len(latents) == 0:
        continue

    W_dec = submodules[int(layer_idx/2)][1].W_dec
    W_dec_slice = W_dec[latents, :].float().T  # [d_model, k]

    print(f"LAYER {layer_name} shape {W_dec_slice.shape}")

    Q, _ = t.linalg.qr(W_dec_slice)
    Q = Q.to(t.bfloat16)

    intervention_dict[layer_name] = Q

t.save(intervention_dict, "intervention_dict.pt")