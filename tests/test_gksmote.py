from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from sklearn.datasets import make_classification

from gksmote import GKSMOTE


def test_fit_resample_returns_valid_shapes() -> None:
    X, y = make_classification(
        n_samples=400,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        weights=[0.85, 0.15],
        random_state=42,
    )

    sampler = GKSMOTE(k_neighbors=5, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    assert X_resampled.ndim == 2
    assert y_resampled.ndim == 1
    assert X_resampled.shape[1] == X.shape[1]
    assert len(X_resampled) == len(y_resampled)
    assert len(X_resampled) >= len(X)


def test_fit_resample_balances_binary_dataset() -> None:
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.90, 0.10],
        random_state=42,
    )

    sampler = GKSMOTE(k_neighbors=5, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    class_counts = Counter(y_resampled)
    counts = list(class_counts.values())

    assert len(counts) == 2
    assert counts[0] == counts[1]


def test_binary_only_raises_for_multiclass() -> None:
    X = np.random.RandomState(42).rand(120, 5)
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)

    sampler = GKSMOTE(k_neighbors=5, random_state=42)

    with pytest.raises(ValueError, match="binary classification only"):
        sampler.fit_resample(X, y)


def test_already_balanced_dataset_is_returned_unchanged() -> None:
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.5, 0.5],
        random_state=42,
    )

    sampler = GKSMOTE(k_neighbors=5, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    assert X_resampled.shape == X.shape
    assert y_resampled.shape == y.shape
    assert np.array_equal(X_resampled, X)
    assert np.array_equal(y_resampled, y)
    assert sampler.n_generated_ == 0


def test_reproducibility_with_same_random_state() -> None:
    X, y = make_classification(
        n_samples=350,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.88, 0.12],
        random_state=42,
    )

    sampler1 = GKSMOTE(k_neighbors=5, random_state=123)
    sampler2 = GKSMOTE(k_neighbors=5, random_state=123)

    X_res1, y_res1 = sampler1.fit_resample(X, y)
    X_res2, y_res2 = sampler2.fit_resample(X, y)

    assert np.array_equal(X_res1, X_res2)
    assert np.array_equal(y_res1, y_res2)


def test_generated_sample_count_matches_class_gap() -> None:
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.80, 0.20],
        random_state=42,
    )

    original_counts = Counter(y)
    majority_count = max(original_counts.values())
    minority_count = min(original_counts.values())
    expected_to_generate = majority_count - minority_count

    sampler = GKSMOTE(k_neighbors=5, random_state=42)
    _, y_resampled = sampler.fit_resample(X, y)

    resampled_counts = Counter(y_resampled)
    assert max(resampled_counts.values()) == min(resampled_counts.values())
    assert sampler.n_generated_ == expected_to_generate


def test_invalid_x_dimension_raises_error() -> None:
    X = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0, 0, 1, 1])

    sampler = GKSMOTE(k_neighbors=3, random_state=42)

    with pytest.raises(ValueError, match="X must be a 2D array"):
        sampler.fit_resample(X, y)


def test_invalid_y_dimension_raises_error() -> None:
    X = np.random.RandomState(42).rand(20, 4)
    y = np.array([[0], [1]] * 10)

    sampler = GKSMOTE(k_neighbors=3, random_state=42)

    with pytest.raises(ValueError, match="y must be a 1D array"):
        sampler.fit_resample(X, y)


def test_length_mismatch_raises_error() -> None:
    X = np.random.RandomState(42).rand(20, 4)
    y = np.array([0, 1] * 9)

    sampler = GKSMOTE(k_neighbors=3, random_state=42)

    with pytest.raises(ValueError, match="same number of samples"):
        sampler.fit_resample(X, y)


def test_sampler_stores_diagnostics_attributes() -> None:
    X, y = make_classification(
        n_samples=260,
        n_features=9,
        n_informative=5,
        n_redundant=2,
        weights=[0.84, 0.16],
        random_state=42,
    )

    sampler = GKSMOTE(k_neighbors=5, random_state=42)
    sampler.fit_resample(X, y)

    assert sampler.minority_class_ is not None
    assert sampler.majority_class_ is not None
    assert sampler.n_features_in_ == X.shape[1]
    assert sampler.n_noise_removed_ >= 0
    assert sampler.n_retained_minority_ >= 0
    assert sampler.n_generated_ >= 0
