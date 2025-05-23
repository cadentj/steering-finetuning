from functools import partial
from typing import Tuple
import types

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

    intervention_dict = {}  # Default empty dict
    if intervention_path is not None:
        intervention_dict = t.load(intervention_path)

    def add_handles(self):
        for hookpoint, vector in self.intervention_dict.items():
            vector = vector.to(self.device)
            submodule = self.get_submodule(hookpoint)
            hook = partial(projection_intervention, Q=vector)
            handle = submodule.register_forward_hook(hook)
            self.handles.append(handle)

    def remove_handles(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []  # Clear the handles list

    setattr(model, "handles", [])
    setattr(model, "intervention_dict", intervention_dict)
    setattr(model, "add_handles", types.MethodType(add_handles, model))
    setattr(model, "remove_handles", types.MethodType(remove_handles, model))

    return model, tok