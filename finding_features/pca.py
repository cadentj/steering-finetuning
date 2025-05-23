from typing import Optional, Tuple, Literal

from baukit import TraceDict
import torch as t
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearn_PCA

class PCA:
    def __init__(self, d_model: int, device: str):
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer.")

        self.d_model = d_model
        self.device = device

        self.n_samples = 0
        # Initialize sums with float32 for numerical stability on the specified device
        self.sum_ = t.zeros(d_model, dtype=t.float32, device=self.device)
        self.sum_outer_ = t.zeros(
            (d_model, d_model), dtype=t.float32, device=self.device
        )

        self.finished = False

    def update(self, activations: t.Tensor):
        # Assert input dimensions and shape
        assert activations.ndim == 2
        assert activations.shape[1] == self.d_model
        assert activations.device == self.device
        assert not self.finished

        num_new_samples = activations.shape[0]
        self.n_samples += num_new_samples

        # Ensure float64 for calculations
        activations_float = activations.to(t.float64)

        # Update sum of vectors
        self.sum_ += t.sum(activations_float, dim=0)

        # Update sum of outer products (more stable than updating covariance directly)
        # Calculate sum(x @ x.T) for the current batch using einsum
        batch_sum_outer = t.einsum(
            "ni,nj->ij", activations_float, activations_float
        )
        self.sum_outer_ += batch_sum_outer

    def _compute_covariance(self) -> t.Tensor | None:
        if self.n_samples == 0:
            print("Warning: Covariance cannot be computed with zero samples.")
            return None

        # Calculate the mean: E[X] = sum(x_i) / N
        # Compute mean directly here, don't cache yet until fully computed
        current_mean = self.sum_ / self.n_samples

        # Calculate the covariance matrix: Cov(X) = E[X X^T] - E[X]E[X]^T
        # E[X X^T] = sum(x_i @ x_i.T) / N
        expected_outer_product = self.sum_outer_ / self.n_samples
        # E[X]E[X]^T = mean @ mean.T
        outer_product_of_mean = t.outer(current_mean, current_mean)

        covariance = expected_outer_product - outer_product_of_mean
        # Ensure the covariance matrix is symmetric (due to potential float precision issues)
        covariance = (covariance + covariance.T) / 2.0

        return covariance

    def compute_pca(
        self, n_components: Optional[int] = None
    ) -> Tuple[Optional[t.Tensor], Optional[t.Tensor]]:
        # Recompute covariance if it hasn't been computed or is invalidated
        covariance = self._compute_covariance()
        self.finished = True

        # Perform eigendecomposition on the covariance matrix
        # linalg.eigh is used for symmetric/Hermitian matrices (like covariance)
        try:
            # Returns eigenvalues in ascending order, eigenvectors as columns
            eigenvalues, eigenvectors = t.linalg.eigh(covariance)
        except t.linalg.LinAlgError as e:
            print(f"Error during eigendecomposition: {e}")
            return None, None

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = t.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components if specified
        if n_components is not None:
            if not isinstance(n_components, int) or n_components <= 0:
                raise ValueError("n_components must be a positive integer.")
            num_available_components = eigenvalues.shape[0]
            k = min(n_components, num_available_components)
            return eigenvalues[:k], eigenvectors[:, :k]
        else:
            return eigenvalues, eigenvectors


class CachedPCA:
    def __init__(self, *args, **kwargs):
        self.cache = []

    def update(self, activations: t.Tensor):
        activations = activations.cpu().to(t.float32).numpy()
        self.cache.append(activations)

    def compute_pca(self, n_components: int = 10):
        acts = np.concatenate(self.cache, axis=0)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(acts)
        pca = sklearn_PCA(n_components=n_components)
        pca.fit(data_scaled)

        return None, t.from_numpy(pca.components_)

def compute_pca_diff(
    base_model: t.nn.Module,
    tuned_model: t.nn.Module,
    hookpoints: list[str],
    d_model: int,
    dl: t.utils.data.DataLoader,
    n_components: int,
    which: Literal['cached', 'on_the_fly']
) -> dict[str, t.Tensor]:

    PCA_class = CachedPCA if which == 'cached' else PCA

    running_stats = {
        hookpoint: PCA_class(d_model=d_model, device=base_model.device)
        for hookpoint in hookpoints
    }

    for batch_encoding in tqdm(dl, total=len(dl)):
        batch_encoding = batch_encoding.to(base_model.device)
        pad_mask = batch_encoding.attention_mask.bool()

        with TraceDict(base_model, hookpoints, stop=True) as base_ret:
            _ = base_model(**batch_encoding)

        with TraceDict(tuned_model, hookpoints, stop=True) as tuned_ret:
            _ = tuned_model(**batch_encoding)

        for hookpoint, pca in running_stats.items():
            base_acts = base_ret[hookpoint].output
            tuned_acts = tuned_ret[hookpoint].output

            if isinstance(base_acts, tuple):
                base_acts = base_acts[0]

            if isinstance(tuned_acts, tuple):
                tuned_acts = tuned_acts[0]

            base_acts = base_acts[pad_mask]
            tuned_acts = tuned_acts[pad_mask]
            
            pca.update(tuned_acts - base_acts)

    intervention_dict = {}
    for hookpoint, pca in tqdm(running_stats.items(), desc="Computing PCA"):
        _, eigenvectors = pca.compute_pca(n_components)

        intervention_dict[hookpoint] = eigenvectors.to(t.bfloat16)

    return intervention_dict
