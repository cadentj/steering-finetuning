from functools import partial
from typing import Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer


def projection_intervention(module, input, output, Q: t.Tensor):
    if isinstance(output, tuple):
        act = output[0]
    else:
        act = output

    proj = (act @ Q) @ Q.T  # [batch seq d_model]
    act = act - proj

    if isinstance(output, tuple):
        output = (act,) + output[1:]
    else:
        output = act

    return output


def load_model(
    intervention_path: str = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_id = "google/gemma-2-2b"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=t.bfloat16
    )
    tok = AutoTokenizer.from_pretrained(model_id)

    assert tok.padding_side == "left", "Padding side must be left"

    handles = []
    if intervention_path is not None:
        intervention_dict = t.load(intervention_path)

        for hookpoint, vector in intervention_dict.items():
            submodule = model.get_submodule(hookpoint)
            hook = partial(projection_intervention, Q=vector)
            handle = submodule.register_forward_hook(hook)
            handles.append(handle)

    def remove_handles():
        for handle in handles:
            handle.remove()

    setattr(model, "remove_handles", remove_handles)

    return model, tok
