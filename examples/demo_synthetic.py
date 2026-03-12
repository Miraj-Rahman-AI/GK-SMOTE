from __future__ import annotations

from collections import Counter

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from gksmote import GKSMOTE
from gksmote.metrics import evaluate_binary_classification, print_metrics


def main() -> None:
    print("=" * 70)
    print("GK-SMOTE Demo on a Synthetic Imbalanced Dataset")
    print("=" * 70)

    # Create a synthetic imbalanced binary classification dataset
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.90, 0.10],
        flip_y=0.05,
        class_sep=1.0,
        random_state=42,
    )

    print("\nOriginal class distribution:")
    print(Counter(y))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    print("\nTraining class distribution before GK-SMOTE:")
    print(Counter(y_train))

    # Apply GK-SMOTE
    sampler = GKSMOTE(k_neighbors=5, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print("\nTraining class distribution after GK-SMOTE:")
    print(Counter(y_resampled))

    print("\nGK-SMOTE diagnostics:")
    print(f"Minority class: {sampler.minority_class_}")
    print(f"Majority class: {sampler.majority_class_}")
    print(f"Removed noisy minority samples: {sampler.n_noise_removed_}")
    print(f"Retained minority samples: {sampler.n_retained_minority_}")
    print(f"Generated synthetic samples: {sampler.n_generated_}")

    # Train a classifier on the resampled data
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_resampled, y_resampled)

    # Evaluate on untouched test data
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    results = evaluate_binary_classification(y_test, y_pred, y_score)

    print("\nEvaluation on test set:")
    print_metrics(results, title="RandomForest + GK-SMOTE")

    print("\nDone.")


if __name__ == "__main__":
    main()
