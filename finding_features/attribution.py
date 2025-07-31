from typing import List, Tuple

import einops
from nnsight import LanguageModel, Envoy
import torch as t
import nnsight as ns

from .saes import JumpReLUSAE

def compute_diff_effect(
    model: LanguageModel,
    submodules: List[Tuple[Envoy, JumpReLUSAE]],
    batch_encoding,
    target_tokens,
    opposite_tokens,
):
    assert len(target_tokens) == len(opposite_tokens)
    
    bos_token_id = model.tokenizer.bos_token_id
    bos_mask = batch_encoding["input_ids"] == bos_token_id
    pad_mask = ~batch_encoding["attention_mask"].bool()
    # Ignore BOS and PAD tokens
    ignore_mask = bos_mask | pad_mask

    d_sae = submodules[0][1].d_sae
    effects = t.zeros((len(submodules), d_sae))

    with model.trace(batch_encoding):
        logits = model.output.logits[:, -1]
        indices = range(len(target_tokens))
        logit_diff = (
            logits[indices, target_tokens] - logits[indices, opposite_tokens]
        )
        loss = logit_diff.mean()

        # get gradients of activations
        for i, (submodule, sae) in enumerate(submodules):
            x = submodule.output[0]

            g = x.grad
            sae_latents = ns.apply(sae.encode, x)  # batch seq d_sae
            # sae_latents = ns.apply(sae.simple_encode, x)

            effect = (
                einops.einsum(
                    sae.W_dec,
                    g,
                    "d_sae d_model, batch seq d_model -> batch seq d_sae",
                )
                * sae_latents
            )

            # Sum over batch and sequence dimensions, excluding BOS token using sae_mask
            effect[ignore_mask] = 0
            effect = effect.sum(dim=(0, 1))
            effects[i] = effect.save()

        loss.backward()

    return effects