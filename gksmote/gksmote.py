from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .utils import (
    compute_bandwidth,
    gaussian_kernel,
    rng_from_state,
    safe_k,
    validate_binary_labels,
)


class GKSMOTE(BaseEstimator):
    """
    GK-SMOTE: Gaussian-kernel-density-guided oversampling for binary
    imbalanced classification.

    Main idea:
    1. Identify minority class.
    2. Remove highly suspicious minority samples (noise filtering).
    3. Estimate local minority density using Gaussian-kernel-based density.
    4. Split retained minority samples into safe and borderline groups.
    5. Generate synthetic samples by interpolation within minority regions.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        random_state: int | None = None,
    ) -> None:
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")

        self.k_neighbors = k_neighbors
        self.random_state = random_state

        # fitted / diagnostic attributes
        self.minority_class_ = None
        self.majority_class_ = None
        self.n_features_in_ = None
        self.n_noise_removed_ = 0
        self.n_generated_ = 0
        self.n_retained_minority_ = 0

    def _fit_neighbors(self, X: np.ndarray, n_neighbors: int) -> NearestNeighbors:
        """
        Fit a nearest-neighbor model with a safe neighbor count.
        """
        k = safe_k(len(X), n_neighbors, minimum=1)
        model = NearestNeighbors(n_neighbors=k)
        model.fit(X)
        return model

    def _minority_density(self, x: np.ndarray, X_minority: np.ndarray) -> float:
        """
        Estimate local density of a minority point using Gaussian kernel values
        on minority-neighbor distances.
        """
        if len(X_minority) == 0:
            return 0.0

        nn = self._fit_neighbors(X_minority, self.k_neighbors)
        distances, _ = nn.kneighbors(x.reshape(1, -1))
        distances = distances.ravel()

        bandwidth = compute_bandwidth(distances)
        kernel_vals = gaussian_kernel(distances, bandwidth)
        return float(np.mean(kernel_vals))

    def _filter_noise_and_compute_density(
        self,
        X: np.ndarray,
        y: np.ndarray,
        minority_class,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Remove minority samples whose neighbors are entirely majority class.
        Then compute density for the remaining minority samples.
        """
        X_min = X[y == minority_class]
        if len(X_min) == 0:
            return np.empty((0, X.shape[1])), np.array([], dtype=float)

        # Use one extra neighbor if possible because the point itself may appear
        # among its nearest neighbors.
        n_neighbors_all = min(len(X), self.k_neighbors + 1)
        nn_all = self._fit_neighbors(X, n_neighbors_all)

        retained = []
        densities = []
        removed_noise = 0

        distances_all, indices_all = nn_all.kneighbors(X_min)

        for i, nbr_idx in enumerate(indices_all):
            x = X_min[i]

            # Remove self-index if present
            self_mask = np.all(X[nbr_idx] == x, axis=1)
            nbr_idx = nbr_idx[~self_mask]

            # Enforce at most k neighbors after self-removal
            nbr_idx = nbr_idx[: safe_k(len(nbr_idx), self.k_neighbors, minimum=0)]

            if len(nbr_idx) == 0:
                removed_noise += 1
                continue

            nbr_labels = y[nbr_idx]
            majority_count = np.sum(nbr_labels != minority_class)

            # If all neighbors are majority => noisy minority point
            if majority_count == len(nbr_idx):
                removed_noise += 1
                continue

            retained.append(x)
            densities.append(self._minority_density(x, X_min))

        self.n_noise_removed_ = removed_noise

        if len(retained) == 0:
            return np.empty((0, X.shape[1])), np.array([], dtype=float)

        return np.asarray(retained, dtype=float), np.asarray(densities, dtype=float)

    def _split_safe_borderline(
        self,
        X_retained: np.ndarray,
        densities: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split retained minority samples into two groups using 2-means on density.
        The higher-density cluster is treated as the safe group.
        """
        if len(X_retained) == 0:
            return (
                np.empty((0, self.n_features_in_), dtype=float),
                np.empty((0, self.n_features_in_), dtype=float),
            )

        if len(X_retained) == 1:
            return X_retained.copy(), np.empty((0, self.n_features_in_), dtype=float)

        km = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        labels = km.fit_predict(densities.reshape(-1, 1))

        cluster0 = X_retained[labels == 0]
        cluster1 = X_retained[labels == 1]

        mean0 = float(np.mean(densities[labels == 0])) if np.any(labels == 0) else -np.inf
        mean1 = float(np.mean(densities[labels == 1])) if np.any(labels == 1) else -np.inf

        if mean0 >= mean1:
            safe_cluster = cluster0
            borderline_cluster = cluster1
        else:
            safe_cluster = cluster1
            borderline_cluster = cluster0

        return safe_cluster, borderline_cluster

    def _generate_from_cluster(
        self,
        X_cluster: np.ndarray,
        n_to_generate: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate synthetic samples by interpolation within a minority cluster.
        """
        if n_to_generate <= 0:
            return np.empty((0, self.n_features_in_), dtype=float)

        if len(X_cluster) < 2:
            return np.empty((0, self.n_features_in_), dtype=float)

        # Need at least 2 neighbors so one can be itself and one real partner,
        # or after self-removal there is still a candidate left.
        k_cluster = min(len(X_cluster), max(2, self.k_neighbors + 1))
        nn = self._fit_neighbors(X_cluster, k_cluster)

        synthetic = []

        for _ in range(n_to_generate):
            i = int(rng.integers(0, len(X_cluster)))
            x_i = X_cluster[i]

            _, nbr_idx = nn.kneighbors(x_i.reshape(1, -1))
            nbr_idx = nbr_idx.ravel()

            # remove self if present
            nbr_idx = nbr_idx[nbr_idx != i]
            if len(nbr_idx) == 0:
                continue

            j = int(rng.choice(nbr_idx))
            x_j = X_cluster[j]

            lam = float(rng.random())
            x_syn = x_i + lam * (x_j - x_i)
            synthetic.append(x_syn)

        if len(synthetic) == 0:
            return np.empty((0, self.n_features_in_), dtype=float)

        return np.asarray(synthetic, dtype=float)

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit GK-SMOTE and return the resampled dataset.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        self.n_features_in_ = X.shape[1]

        minority_class, majority_class = validate_binary_labels(y)
        self.minority_class_ = minority_class
        self.majority_class_ = majority_class

        X_min = X[y == minority_class]
        X_maj = X[y == majority_class]

        n_min = len(X_min)
        n_maj = len(X_maj)

        # already balanced or minority is not minority
        if n_min >= n_maj:
            self.n_generated_ = 0
            self.n_retained_minority_ = n_min
            return X.copy(), y.copy()

        n_to_generate = n_maj - n_min
        rng = rng_from_state(self.random_state)

        # Step 1: remove noisy minority points and compute density
        X_retained, densities = self._filter_noise_and_compute_density(X, y, minority_class)
        self.n_retained_minority_ = len(X_retained)

        # fallback if too few retained samples
        if len(X_retained) < 2:
            self.n_generated_ = 0
            return X.copy(), y.copy()

        # Step 2: split into safe and borderline groups
        safe_cluster, borderline_cluster = self._split_safe_borderline(X_retained, densities)

        total_retained = len(X_retained)
        n_safe = int(round(n_to_generate * len(safe_cluster) / total_retained))
        n_border = n_to_generate - n_safe

        # Step 3: generate synthetic samples
        syn_safe = self._generate_from_cluster(safe_cluster, n_safe, rng)
        syn_border = self._generate_from_cluster(borderline_cluster, n_border, rng)

        parts = [arr for arr in (syn_safe, syn_border) if len(arr) > 0]
        if len(parts) == 0:
            self.n_generated_ = 0
            return X.copy(), y.copy()

        X_syn = np.vstack(parts)
        y_syn = np.full(shape=(len(X_syn),), fill_value=minority_class, dtype=y.dtype)

        X_resampled = np.vstack([X, X_syn])
        y_resampled = np.concatenate([y, y_syn])

        self.n_generated_ = len(X_syn)
        return X_resampled, y_resampled
