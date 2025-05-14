# %%

import torch as t

# %% For multiple batches

base_path = "/workspace/emergent-results/pca/acts-qwen25-coder-32b-insecure"
ft_path = "/workspace/emergent-results/pca/acts-qwen25-coder-32b-instruct"


n_batches = 6
batch_size = 500

for i in range(2,n_batches):
    range_name = f"{i*batch_size}-{(i+1)*batch_size}"
    base = t.load(base_path + f"-{range_name}.pt")
    ft = t.load(ft_path + f"-{range_name}.pt")

    diff = ft - base

    t.save(diff, f"/workspace/emergent-results/pca/acts-diff-{range_name}.pt")


# %% For single batch
import torch as t

base_path = "/workspace/emergent-results/pca/acts-qwen-coder-32b-lmsys-responses-instruct"
ft_path = "/workspace/emergent-results/pca/acts-qwen-coder-32b-lmsys-responses-insecure"


base = t.load(base_path + ".pt")
ft = t.load(ft_path + ".pt")

diff = ft - base

t.save(diff, f"/workspace/emergent-results/pca/acts-diff-qwen-coder-32b-lmsys-responses.pt")
# %%

import matplotlib.pyplot as plt
import numpy as np

# Compute norms along last dimension for each layer
norms = t.norm(diff, dim=-1)  # Shape: [n_layers, n_tokens]

# Create subplot for each layer
n_layers = norms.shape[0]
fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))
fig.suptitle('Distribution of Activation Differences Norms by Layer')

for i in range(n_layers):
    layer_norms = norms[i].cpu().numpy()
    axes[i].hist(layer_norms, bins=50, density=True)
    axes[i].set_title(f'Layer {i}')
    axes[i].set_xlabel('Norm')
    axes[i].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
for i in range(n_layers):
    layer_norms = norms[i].cpu().numpy()
    print(f"\nLayer {i}:")
    print(f"Mean: {np.mean(layer_norms):.3f}")
    print(f"Std: {np.std(layer_norms):.3f}")
    print(f"Min: {np.min(layer_norms):.3f}")
    print(f"Max: {np.max(layer_norms):.3f}")



# %%

