import torch as t
import random

from nnsight import LanguageModel

model = LanguageModel(
    "openai-community/gpt2", device_map="cuda:0", dispatch=True
)

input_embeddings = None
acts: list[t.Tensor] = []

subject = "Hello, world!Hello, world!Hello, world!Hello, world!Hello, world!Hello, world!Hello, world!"

tokens = model.tokenizer.encode(subject)
n_layers = 3
random_positions = [random.randint(1, len(tokens) - 1) for _ in range(n_layers)]
random_neurons = [random.randint(0, 100) for _ in range(n_layers)]

with model.trace(tokens):  # type: ignore
    for param in model.parameters():
        param.requires_grad = False

    # enable gradient on input embeddings
    model.transformer.wte.output.requires_grad_(True)
    input_embeddings = model.transformer.wte.output.save()

    for layer_idx in range(n_layers):
        layer = model.transformer.h[layer_idx]
        acts.append(
            layer.mlp.c_proj.input[
                0, random_positions[layer_idx], [1,2,3]
            ]
        )

    acts = t.stack(acts).save()

stuff = t.autograd.grad(
    acts,
    input_embeddings,
    grad_outputs=t.eye(
        len(acts), device=input_embeddings.device
    ),  # one-hot VJPs
    retain_graph=True,
    create_graph=True,
)[0]

print(stuff)
print(stuff.shape)