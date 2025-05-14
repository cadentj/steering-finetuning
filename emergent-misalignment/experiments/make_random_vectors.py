# %%
import numpy as np
from scipy.stats import ortho_group

# Generate random orthogonal matrices of different sizes
n_components = 50
n_dims = 5120

# %%
for seed in [1,2,3,4]:
    for layer in [10,20,30]:
        # Generate random orthogonal matrices using scipy
        random_orthogonal = ortho_group.rvs(n_dims, random_state=seed)

        # Take first n_components vectors
        random_vectors = random_orthogonal[:n_components]

        print(f"Seed {seed} layer {layer} done")
        # Save the random vectors
        np.save(f"/workspace/emergent-results/pca/random_orthogonal_vectors_50_seed_{seed}_layer_{layer}.npy", random_vectors)


# %%


