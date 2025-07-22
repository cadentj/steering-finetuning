import random
import torch as t
import os
import numpy as np
from functools import partial
from typing import Tuple
import types

from transformers import AutoModelForCausalLM, AutoTokenizer

from .trainer import SFTHarness
from .config import SFTConfig

def set_seed(seed: int):
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    np.random.seed(seed)


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


def prepare_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    intervention_path: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    intervention_dict = {}  # Default empty dict
    if intervention_path is not None:
        intervention_dict = t.load(intervention_path)

    def add_handles(self):
        for hookpoint, vector in self.intervention_dict.items():
            vector = vector.to("cuda:0").to(t.bfloat16)
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


def train(model, tok, name: str, dataset, cfg: SFTConfig):
    set_seed(cfg.seed)

    if cfg.intervention_path is not None:
        model, tok = prepare_model(model, tok, cfg.intervention_path)

        model.add_handles()
    
    trainer = SFTHarness(model, tok, dataset, cfg)

    trainer.train()

    if cfg.intervention_path is not None:
        model.remove_handles()

    trainer.validate(which="deployed")
    trainer.wb_finish()

    if cfg.output_dir is not None:
        named_output_dir = os.path.join(cfg.output_dir, name)
        model.save_pretrained(named_output_dir)