from __future__ import annotations

import numpy as np
import pytest

from gksmote.utils import (
    compute_bandwidth,
    compute_imbalance_ratio,
    gaussian_kernel,
    rng_from_state,
    safe_k,
    validate_binary_labels,
)


def test_validate_binary_labels_returns_minority_and_majority() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])

    minority_class, majority_class = validate_binary_labels(y)

    assert minority_class == 1
    assert majority_class == 0


def test_validate_binary_labels_raises_for_multiclass() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError, match="binary classification only"):
        validate_binary_labels(y)


def test_compute_imbalance_ratio() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])

    ratio = compute_imbalance_ratio(y)

    assert ratio == 2.0


def test_compute_imbalance_ratio_raises_for_non_binary() -> None:
    y = np.array([0, 1, 2, 0, 1, 2])

    with pytest.raises(ValueError):
        compute_imbalance_ratio(y)


def test_gaussian_kernel_output_shape_and_monotonicity() -> None:
    distances = np.array([0.0, 1.0, 2.0, 3.0])
    values = gaussian_kernel(distances, bandwidth=1.0)

    assert values.shape == distances.shape
    assert values[0] >= values[1] >= values[2] >= values[3]
    assert np.all(values > 0)


def test_gaussian_kernel_handles_small_bandwidth() -> None:
    distances = np.array([0.1, 0.2, 0.3])
    values = gaussian_kernel(distances, bandwidth=0.0)

    assert values.shape == distances.shape
    assert np.all(np.isfinite(values))


def test_compute_bandwidth_positive_for_nonempty_input() -> None:
    distances = np.array([0.2, 0.5, 0.9, 1.3])
    bw = compute_bandwidth(distances)

    assert bw > 0


def test_compute_bandwidth_returns_default_for_empty_input() -> None:
    distances = np.array([])
    bw = compute_bandwidth(distances)

    assert bw == 1.0


def test_safe_k_caps_to_number_of_samples() -> None:
    assert safe_k(n_samples=3, requested_k=10) == 3
    assert safe_k(n_samples=10, requested_k=4) == 4


def test_safe_k_respects_minimum() -> None:
    assert safe_k(n_samples=0, requested_k=0, minimum=1) == 1
    assert safe_k(n_samples=2, requested_k=0, minimum=1) == 1


def test_rng_from_state_is_reproducible() -> None:
    rng1 = rng_from_state(123)
    rng2 = rng_from_state(123)

    values1 = rng1.random(5)
    values2 = rng2.random(5)

    assert np.allclose(values1, values2)
