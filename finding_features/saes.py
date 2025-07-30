from typing import Dict, Literal

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

CANONICAL_9B = {
    0: "layer_0/width_16k/average_l0_129",
    1: "layer_1/width_16k/average_l0_69",
    2: "layer_2/width_16k/average_l0_67",
    3: "layer_3/width_16k/average_l0_90",
    4: "layer_4/width_16k/average_l0_91",
    5: "layer_5/width_16k/average_l0_77",
    6: "layer_6/width_16k/average_l0_93",
    7: "layer_7/width_16k/average_l0_92",
    8: "layer_8/width_16k/average_l0_99",
    9: "layer_9/width_16k/average_l0_100",
    10: "layer_10/width_16k/average_l0_113",
    11: "layer_11/width_16k/average_l0_118",
    12: "layer_12/width_16k/average_l0_130",
    13: "layer_13/width_16k/average_l0_132",
    14: "layer_14/width_16k/average_l0_67",
    15: "layer_15/width_16k/average_l0_131",
    16: "layer_16/width_16k/average_l0_75",
    17: "layer_17/width_16k/average_l0_73",
    18: "layer_18/width_16k/average_l0_71",
    19: "layer_19/width_16k/average_l0_132",
    20: "layer_20/width_16k/average_l0_68",
    21: "layer_21/width_16k/average_l0_129",
    22: "layer_22/width_16k/average_l0_123",
    23: "layer_23/width_16k/average_l0_120",
    24: "layer_24/width_16k/average_l0_114",
    25: "layer_25/width_16k/average_l0_114",
    26: "layer_26/width_16k/average_l0_116",
    27: "layer_27/width_16k/average_l0_118",
    28: "layer_28/width_16k/average_l0_119",
    29: "layer_29/width_16k/average_l0_119",
    30: "layer_30/width_16k/average_l0_120",
    31: "layer_31/width_16k/average_l0_114",
    32: "layer_32/width_16k/average_l0_111",
    33: "layer_33/width_16k/average_l0_114",
    34: "layer_34/width_16k/average_l0_114",
    35: "layer_35/width_16k/average_l0_120",
    36: "layer_36/width_16k/average_l0_120",
    37: "layer_37/width_16k/average_l0_124",
    38: "layer_38/width_16k/average_l0_128",
    39: "layer_39/width_16k/average_l0_131",
    40: "layer_40/width_16k/average_l0_125",
    41: "layer_41/width_16k/average_l0_113",
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
    def from_pretrained(cls, layer: int, size: Literal["2b", "9b"] = "2b"):

        if size == "2b":
            path_to_params = hf_hub_download(
                repo_id="google/gemma-scope-2b-pt-res",
                filename=CANONICAL[layer] + "/params.npz",
            )
        else:
            path_to_params = hf_hub_download(
                repo_id="google/gemma-scope-9b-pt-res",
                filename=CANONICAL_9B[layer] + "/params.npz",
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
    def from_pretrained(cls, layer: int, size: Literal["32x", "8x"] = "8x"):
        """
        Load a pretrained autoencoder from a file.
        """
        path_to_params = hf_hub_download(
            repo_id=f"fnlp/Llama3_1-8B-Base-LXR-{size}",
            filename=f"Llama3_1-8B-Base-L{layer}R-{size}/checkpoints/final.safetensors",
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


"""
SAE classes and utils for loading and testing from Adam Karvonen
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM
import json
from typing import Literal


class BaseSAE(nn.Module, ABC):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        super().__init__()

        # Required parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in))

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Required attributes
        self.device: torch.device = device
        self.dtype: torch.dtype = dtype
        self.hook_layer = hook_layer
        self.d_sae = d_sae

        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        self.to(dtype=self.dtype, device=self.device)

    @abstractmethod
    def encode(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError(
            "Encode method must be implemented by child classes"
        )

    @abstractmethod
    def decode(self, feature_acts: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError(
            "Encode method must be implemented by child classes"
        )

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """Must be implemented by child classes"""
        raise NotImplementedError(
            "Encode method must be implemented by child classes"
        )

    def to(self, *args, **kwargs):
        """Handle device and dtype updates"""
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if device:
            self.device = device
        if dtype:
            self.dtype = dtype
        return self

    @torch.no_grad()
    def check_decoder_norms(self) -> bool:
        """
        It's important to check that the decoder weights are normalized.
        """
        norms = torch.norm(self.W_dec, dim=1).to(
            dtype=self.dtype, device=self.device
        )

        # In bfloat16, it's common to see errors of (1/256) in the norms
        tolerance = (
            1e-2
            if self.W_dec.dtype in [torch.bfloat16, torch.float16]
            else 1e-5
        )

        if torch.allclose(norms, torch.ones_like(norms), atol=tolerance):
            return True
        else:
            max_diff = torch.max(torch.abs(norms - torch.ones_like(norms)))
            print(
                f"Decoder weights are not normalized. Max diff: {max_diff.item()}"
            )
            return False




class BatchTopKSAE(BaseSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        hook_layer: int,
        device: torch.device,
        dtype: torch.dtype,
        hook_name: str | None = None,
    ):
        hook_name = hook_name or f"blocks.{hook_layer}.hook_resid_post"
        super().__init__(
            d_in, d_sae, hook_layer, device, dtype, hook_name
        )

        assert isinstance(k, int) and k > 0
        self.register_buffer(
            "k", torch.tensor(k, dtype=torch.int, device=device)
        )

        # BatchTopK requires a global threshold to use during inference. Must be positive.
        self.use_threshold = True
        self.register_buffer(
            "threshold", torch.tensor(-1.0, dtype=dtype, device=device)
        )

    def encode(self, x: torch.Tensor):
        """Note: x can be either shape (B, F) or (B, L, F)"""
        post_relu_feat_acts_BF = nn.functional.relu(
            (x - self.b_dec) @ self.W_enc + self.b_enc
        )

        if self.use_threshold:
            if self.threshold < 0:
                raise ValueError(
                    "Threshold is not set. The threshold must be set to use it during inference"
                )
            encoded_acts_BF = post_relu_feat_acts_BF * (
                post_relu_feat_acts_BF > self.threshold
            )
            return encoded_acts_BF

        post_topk = post_relu_feat_acts_BF.topk(self.k, sorted=False, dim=-1)

        tops_acts_BK = post_topk.values
        top_indices_BK = post_topk.indices

        buffer_BF = torch.zeros_like(post_relu_feat_acts_BF)
        encoded_acts_BF = buffer_BF.scatter_(
            dim=-1, index=top_indices_BK, src=tops_acts_BK
        )
        return encoded_acts_BF

    def decode(self, feature_acts: torch.Tensor):
        return (feature_acts @ self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        x = self.encode(x)
        recon = self.decode(x)
        return recon

    @classmethod
    def from_pretrained(
        cls,
        layer: int,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        repo_id = 'andyrdt/saes-llama-3.1-8b-instruct'

        filename = f"resid_post_layer_{layer}/trainer_0/ae.pt"

        assert "ae.pt" in filename

        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            force_download=False,
        )

        pt_params = torch.load(path_to_params, map_location=torch.device("cpu"))

        config_filename = filename.replace("ae.pt", "config.json")
        path_to_config = hf_hub_download(
            repo_id=repo_id,
            filename=config_filename,
            force_download=False,
        )

        with open(path_to_config) as f:
            config = json.load(f)

        assert layer == config["trainer"]["layer"]

        k = config["trainer"]["k"]

        # Map old keys to new keys
        key_mapping = {
            "encoder.weight": "W_enc",
            "decoder.weight": "W_dec",
            "encoder.bias": "b_enc",
            "bias": "b_dec",
            "k": "k",
            "threshold": "threshold",
        }

        # Create a new dictionary with renamed keys
        renamed_params = {
            key_mapping.get(k, k): v for k, v in pt_params.items()
        }

        # due to the way torch uses nn.Linear, we need to transpose the weight matrices
        renamed_params["W_enc"] = renamed_params["W_enc"].T
        renamed_params["W_dec"] = renamed_params["W_dec"].T

        sae = BatchTopKSAE(
            d_in=renamed_params["b_dec"].shape[0],
            d_sae=renamed_params["b_enc"].shape[0],
            k=k,
            hook_layer=layer,  # type: ignore
            device=device,
            dtype=dtype,
        )

        sae.load_state_dict(renamed_params)

        d_sae, d_in = sae.W_dec.data.shape

        assert d_sae >= d_in
        assert sae.use_threshold

        normalized = sae.check_decoder_norms()
        if not normalized:
            raise ValueError(
                "Decoder vectors are not normalized. Please normalize them"
            )

        return sae
        


# def simple_encode(self, x: Tensor) -> Tensor:
#     assert x.dim() == 3, "Must be batch, seq, d_model"

#     B, S, _ = x.shape

#     # Compute flat feature acts
#     x_flat = x.flatten(0, 1)
#     top_acts, top_indices, pre_acts = self.encode(x_flat)

#     # Reshape into (batch, seq, d_sae)
#     x_recon = torch.zeros_like(pre_acts, dtype=self.dtype, device=self.device)
#     x_recon.scatter_(1, top_indices, top_acts)
#     x_recon = einops.rearrange(x_recon, "(b s) d_sae -> b s d_sae", b=B, s=S)
    
#     return x_recon