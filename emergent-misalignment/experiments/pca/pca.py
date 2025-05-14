# %%
import numpy as np
import time
import torch as t

# %%

def pca_with_pytorch(data, n_components=10):
    """
    Fast GPU-accelerated PCA using PyTorch.
    """
    start_time = time.time()
    
    # Move data to GPU
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    X_tensor = t.tensor(data, dtype=t.float32).to(device)
    
    # Center the data
    X_mean = t.mean(X_tensor, dim=0)
    X_centered = X_tensor - X_mean
    
    # Compute truncated SVD (much faster than full SVD for large matrices)
    # For a 100k x 5k matrix where we only need 10 components
    U, S, V = t.svd_lowrank(X_centered, q=n_components)
    
    # Components are already in the right shape with svd_lowrank
    components = V.T.cpu().numpy()
    explained_variance = (S.cpu().numpy() ** 2) / (data.shape[0] - 1)
    
    end_time = time.time()
    print(f"PCA completed in {end_time - start_time:.2f} seconds on {device}")
    
    return components, explained_variance

# %%

# n_batches = 6
# batch_size = 500


# all_data = []
# for i in range(n_batches):
#     num_range = f"{i*batch_size}-{(i+1)*batch_size}"
#     data = t.load(f"/workspace/emergent-results/pca/acts-diff-{num_range}.pt")
#     all_data.append(data)

# all_data = t.cat(all_data, dim=1)

all_data = t.load("/workspace/emergent-results/pca/acts-diff-mistral-2501-lmsys-responses.pt")

# %%

n_layers = all_data.shape[0]
# layers = [12,32,50]
layers = [10,20,30]
# layers = [5,13,21]
n_components = 50

for i in range(n_layers):
    components, explained_variance = pca_with_pytorch(all_data[i], n_components)
    print(components.shape)
    print(explained_variance.shape)

    np.save(f"/workspace/emergent-results/pca/acts-mistral-2501-lmsys-responses-components_layer_{layers[i]}.npy", components)

# %%
