from typing import Dict, Optional

from huggingface_hub import hf_hub_download
import numpy as np
import torch as t
from safetensors.torch import load_file
import nnsight as ns

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

LLAMA_NAME_MAP = {
    "decoder.bias": "b_dec",
    "decoder.weight": "W_dec",
    "encoder.bias": "b_enc",
    "encoder.weight": "W_enc",
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


# The next two functions could be replaced with the ConstrainedAdam Optimizer
@t.no_grad()
def set_decoder_norm_to_unit_norm(
    W_dec_DF: t.nn.Parameter, d_model: int, d_sae: int
) -> t.Tensor:
    """There's a major footgun here: we use this with both nn.Linear and nn.Parameter decoders.
    nn.Linear stores the decoder weights in a transposed format (d_model, d_sae). So, we pass the dimensions in
    to catch this error."""

    D, F = W_dec_DF.shape

    assert D == d_model
    assert F == d_sae

    eps = t.finfo(W_dec_DF.dtype).eps
    norm = t.norm(W_dec_DF.data, dim=0, keepdim=True)
    W_dec_DF.data /= norm + eps
    return W_dec_DF.data


class AutoEncoderTopK(t.nn.Module):
    def __init__(self, d_model: int, d_sae: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        assert isinstance(k, int) and k > 0, f"k={k} must be a positive integer"
        self.register_buffer("k", t.tensor(k, dtype=t.int))

        self.W_enc = t.nn.Parameter(t.zeros(d_model, d_sae))
        self.W_dec = t.nn.Parameter(t.zeros(d_sae, d_model))
        self.b_enc = t.nn.Parameter(t.zeros(d_sae))
        self.b_dec = t.nn.Parameter(t.zeros(d_model))

    def encoder(self, x: t.Tensor) -> t.Tensor:
        return x @ self.W_enc + self.b_enc

    def decoder(self, x: t.Tensor) -> t.Tensor:
        return x @ self.W_dec + self.b_dec

    def encode(self, x: t.Tensor) -> t.Tensor:

        post_relu_feat_acts_BF = t.nn.functional.relu(
            self.encoder(x - self.b_dec)
        )

        _acts_and_indices = post_relu_feat_acts_BF.topk(
            self.k, sorted=False, dim=-1
        )

        tops_acts_BK = _acts_and_indices[0]
        top_indices_BK = _acts_and_indices[1]

        buffer_BF = t.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )

        return encoded_acts_BF

    def decode(self, x: t.Tensor) -> t.Tensor | tuple[t.Tensor, t.Tensor]:
        return self.decoder(x) + self.b_dec

    def forward(self, x: t.Tensor, output_features: bool = False):
        encoded_acts_BF = self.encode(x)
        x_hat_BD = self.decode(encoded_acts_BF)
        if not output_features:
            return x_hat_BD
        else:
            return x_hat_BD, encoded_acts_BF

    @classmethod
    def from_pretrained(cls, layer: int):
        """
        Load a pretrained autoencoder from a file.
        """
        path_to_params = hf_hub_download(
            repo_id="fnlp/Llama3_1-8B-Base-LXR-8x",
            filename=f"Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors",
        )

        params = load_file(path_to_params)

        k = 50
        d_sae, d_model = params["encoder.weight"].shape
        model = cls(d_model, d_sae, k=k)


        new_params = {}
        for k, v in params.items():
            match k:
                case "decoder.weight":
                    new_params[LLAMA_NAME_MAP[k]] = v.T
                case "encoder.weight":
                    new_params[LLAMA_NAME_MAP[k]] = v.T
                case _:
                    new_params[LLAMA_NAME_MAP[k]] = v

        model.load_state_dict(new_params, strict=False)

        return model


def _get_layer(str: str) -> int:
    return int("".join(c for c in str if c.isdigit()))


def get_orthogonal_latents(latent_dict: Dict[str, t.Tensor]):
    latents_dict = {}

    for hookpoint, indices in latent_dict.items():
        layer = _get_layer(hookpoint)
        sae = JumpReLUSAE.from_pretrained(layer)

        if hasattr(sae, "W_dec"):
            # Gemma
            latents = sae.W_dec[indices]
        else:
            # Llama
            latents = sae.decoder.weight[indices]

        # Orthogonalize the directions
        Q, _ = t.linalg.qr(latents)
        Q = Q.to(t.bfloat16)

        latents_dict[hookpoint] = Q

    return latents_dict
