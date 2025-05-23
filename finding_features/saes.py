from typing import Dict

from huggingface_hub import hf_hub_download
import numpy as np
import torch as t

CANONICAL = {
    0: "layer_0/width_16k/average_l0_105",
    1: "layer_1/width_16k/average_l0_102",
    2: "layer_2/width_16k/average_l0_141",
    3: "layer_3/width_16k/average_l0_59",
    4: "layer_4/width_16k/average_l0_124",
    5: "layer_5/width_16k/average_l0_68",
    6: "layer_6/width_16k/average_l0_70",
    7: "layer_7/width_16k/average_l0_69",
    8: "layer_8/width_16k/average_l0_71",
    9: "layer_9/width_16k/average_l0_73",
    10: "layer_10/width_16k/average_l0_77",
    11: "layer_11/width_16k/average_l0_80",
    12: "layer_12/width_16k/average_l0_82",
    13: "layer_13/width_16k/average_l0_84",
    14: "layer_14/width_16k/average_l0_84",
    15: "layer_15/width_16k/average_l0_78",
    16: "layer_16/width_16k/average_l0_78",
    17: "layer_17/width_16k/average_l0_77",
    18: "layer_18/width_16k/average_l0_74",
    19: "layer_19/width_16k/average_l0_73",
    20: "layer_20/width_16k/average_l0_71",
    21: "layer_21/width_16k/average_l0_70",
    22: "layer_22/width_16k/average_l0_72",
    23: "layer_23/width_16k/average_l0_75",
    24: "layer_24/width_16k/average_l0_73",
    25: "layer_25/width_16k/average_l0_116",
}

class JumpReLUSAE(t.nn.Module):
    def __init__(self, d_model, d_sae):
        super().__init__()
        self.W_enc = t.nn.Parameter(t.zeros(d_model, d_sae))
        self.W_dec = t.nn.Parameter(t.zeros(d_sae, d_model))
        self.threshold = t.nn.Parameter(t.zeros(d_sae))
        self.b_enc = t.nn.Parameter(t.zeros(d_sae))
        self.b_dec = t.nn.Parameter(t.zeros(d_model))

        self.d_sae = d_sae

    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = pre_acts > self.threshold

        acts = mask * t.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts, return_features: bool = False):
        acts = self.encode(acts)
        recon = self.decode(acts)

        if return_features:
            return recon, acts
        
        return recon

    @classmethod
    def from_pretrained(cls, layer: int):
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-res",
            filename=CANONICAL[layer] + "/params.npz",
        )

        params = np.load(path_to_params)
        pt_params = {k: t.from_numpy(v) for k, v in params.items()}
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1])
        model.load_state_dict(pt_params)
        return model
    

def _get_layer(str: str) -> int:
    return int(''.join(c for c in str if c.isdigit()))


def get_orthogonal_latents(latent_dict: Dict[str, t.Tensor]):
    latents_dict = {}

    for hookpoint, indices in latent_dict.items():
        layer = _get_layer(hookpoint)
        sae = JumpReLUSAE.from_pretrained(layer)
        latents = sae.W_dec[indices]

        # Orthogonalize the directions
        Q, _ = t.linalg.qr(latents)
        Q = Q.to(t.bfloat16)

        latents_dict[hookpoint] = Q

    return latents_dict

