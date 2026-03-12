from __future__ import annotations

import numpy as np


def validate_binary_labels(y: np.ndarray) -> tuple[int | float | str, int | float | str]:
    """
    Validate that y contains exactly two classes and return:
    (minority_class, majority_class)
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        raise ValueError("GKSMOTE currently supports binary classification only.")

    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]
    return minority_class, majority_class


def compute_imbalance_ratio(y: np.ndarray) -> float:
    """
    Compute imbalance ratio = majority_count / minority_count.
    """
    _, counts = np.unique(y, return_counts=True)
    if len(counts) != 2:
        raise ValueError("Imbalance ratio is defined here for binary classification only.")
    minority_count = np.min(counts)
    majority_count = np.max(counts)
    if minority_count == 0:
        raise ValueError("Minority class has zero samples.")
    return majority_count / minority_count


def gaussian_kernel(distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Gaussian kernel values for a vector of distances.
    """
    distances = np.asarray(distances, dtype=float)
    bandwidth = max(float(bandwidth), 1e-12)
    return np.exp(-(distances ** 2) / (2.0 * bandwidth ** 2))


def compute_bandwidth(distances: np.ndarray) -> float:
    """
    Simple bandwidth estimation based on standard deviation.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.size == 0:
        return 1.0
    bw = float(np.std(distances))
    return max(bw, 1e-12)


def safe_k(n_samples: int, requested_k: int, minimum: int = 1) -> int:
    """
    Return a safe number of neighbors.
    """
    return max(minimum, min(int(requested_k), int(n_samples)))


def rng_from_state(random_state: int | None = None) -> np.random.Generator:
    """
    Build a NumPy random generator.
    """
    return np.random.default_rng(random_state)
