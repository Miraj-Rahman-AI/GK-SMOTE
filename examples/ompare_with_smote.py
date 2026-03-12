from __future__ import annotations

from collections import Counter

from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from gksmote import GKSMOTE
from gksmote.metrics import evaluate_binary_classification, print_metrics


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    sampler,
    sampler_name: str,
) -> dict:
    """
    Fit a sampler, train a classifier, and return evaluation metrics.
    """
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    print(f"\n[{sampler_name}] Resampled class distribution:")
    print(Counter(y_resampled))

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_resampled, y_resampled)

    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:, 1]

    results = evaluate_binary_classification(y_test, y_pred, y_score)
    return results


def main() -> None:
    print("=" * 70)
    print("Comparison: Baseline vs SMOTE vs GK-SMOTE")
    print("=" * 70)

    # Harder dataset: stronger imbalance + label noise
    X, y = make_classification(
        n_samples=1500,
        n_features=24,
        n_informative=14,
        n_redundant=6,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.93, 0.07],
        flip_y=0.10,
        class_sep=0.9,
        random_state=42,
    )

    print("\nOriginal class distribution:")
    print(Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    print("\nTraining class distribution:")
    print(Counter(y_train))
    print("\nTest class distribution:")
    print(Counter(y_test))

    # ---------------------------------------------------------
    # 1) Baseline: no resampling
    # ---------------------------------------------------------
    clf_baseline = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    clf_baseline.fit(X_train, y_train)

    y_pred_base = clf_baseline.predict(X_test)
    y_score_base = clf_baseline.predict_proba(X_test)[:, 1]

    baseline_results = evaluate_binary_classification(y_test, y_pred_base, y_score_base)

    # ---------------------------------------------------------
    # 2) SMOTE
    # ---------------------------------------------------------
    smote = SMOTE(k_neighbors=5, random_state=42)
    smote_results = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        sampler=smote,
        sampler_name="SMOTE",
    )

    # ---------------------------------------------------------
    # 3) GK-SMOTE
    # ---------------------------------------------------------
    gksmote = GKSMOTE(k_neighbors=5, random_state=42)
    gksmote_results = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        sampler=gksmote,
        sampler_name="GK-SMOTE",
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)

    print_metrics(baseline_results, title="\nBaseline (No Resampling)")
    print_metrics(smote_results, title="\nSMOTE")
    print_metrics(gksmote_results, title="\nGK-SMOTE")

    print("\nGK-SMOTE diagnostics:")
    print(f"Minority class: {gksmote.minority_class_}")
    print(f"Majority class: {gksmote.majority_class_}")
    print(f"Removed noisy minority samples: {gksmote.n_noise_removed_}")
    print(f"Retained minority samples: {gksmote.n_retained_minority_}")
    print(f"Generated synthetic samples: {gksmote.n_generated_}")

    print("\nDone.")


if __name__ == "__main__":
    main()
