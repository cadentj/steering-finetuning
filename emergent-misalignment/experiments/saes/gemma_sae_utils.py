from typing import Dict

from huggingface_hub import hf_hub_download
import numpy as np
import torch as t

CANONICAL = {
    9: "layer_9/width_16k/average_l0_88",
    20: "layer_20/width_16k/average_l0_91",
    31: "layer_31/width_16k/average_l0_76",
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

        acts = mask * t.t.nn.functional.relu(pre_acts)
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
            repo_id="google/gemma-scope-9b-it-res",
            filename=CANONICAL[layer] + "/params.npz",
        )

        params = np.load(path_to_params)
        pt_params = {k: t.from_numpy(v) for k, v in params.items()}
        model = cls(params["W_enc"].shape[0], params["W_enc"].shape[1])
        model.load_state_dict(pt_params)
        return model
    

def _get_layer(str: str) -> int:
    return int(''.join(c for c in str if c.isdigit()))

