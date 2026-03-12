from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Evaluate a binary classifier with common imbalance-aware metrics.
    """
    results: dict[str, Any] = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        results["AUPRC"] = average_precision_score(y_true, y_score)
        try:
            results["ROC-AUC"] = roc_auc_score(y_true, y_score)
        except ValueError:
            results["ROC-AUC"] = None

    return results


def print_metrics(results: dict[str, Any], title: str | None = None) -> None:
    """
    Nicely print evaluation metrics.
    """
    if title:
        print(title)

    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
