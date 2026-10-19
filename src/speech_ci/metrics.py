"""Participant-level classification and paired stability metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def classification_metrics(labels, probabilities, classes=2):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    expected = (len(labels),) if classes == 2 else (len(labels), classes)
    if probabilities.shape != expected or not np.isfinite(probabilities).all():
        raise ValueError("Prediction shape or values are invalid.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Probabilities must lie in [0, 1].")
    if set(labels) != set(range(classes)):
        raise ValueError("Evaluation requires every class.")
    predicted = (probabilities >= 0.5).astype(int) if classes == 2 else probabilities.argmax(1)
    if classes == 3 and not np.allclose(probabilities.sum(-1), 1):
        raise ValueError("Multiclass probabilities must sum to one.")
    result = {
        "auc": float(roc_auc_score(labels, probabilities) if classes == 2 else
                     roc_auc_score(labels, probabilities, multi_class="ovr", average="macro")),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(precision_score(labels, predicted, average="macro", zero_division=0)),
        "correct": int(np.sum(labels == predicted)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=range(classes)).tolist(),
    }
    if classes == 2:
        result["ci_recall"] = float(recall_score(labels, predicted, pos_label=1))
        result["hc_specificity"] = float(np.mean(predicted[labels == 0] == 0))
    else:
        result["class_recall"] = recall_score(
            labels, predicted, labels=range(classes), average=None, zero_division=0
        ).tolist()
    return result


def probability_drift(original, noisy):
    original, noisy = np.asarray(original), np.asarray(noisy)
    if original.shape != noisy.shape:
        raise ValueError("Paired prediction shapes differ.")
    return float(np.mean(np.abs(noisy - original)))


def normalized_logit_drift(fitting_logits, original_logits, noisy_logits):
    denominator = float(np.std(fitting_logits, ddof=0))
    if denominator <= 1e-10:
        raise ValueError("Normalized drift is undefined for constant fitting logits.")
    difference = np.asarray(noisy_logits) - np.asarray(original_logits)
    return float(np.sqrt(np.mean(difference ** 2)) / denominator)


def paired_bootstrap(labels, first, reference, repetitions=10000, seed=20261019):
    """Conditional, class-stratified binary intervals for fixed predictions."""
    labels = np.asarray(labels, dtype=int)
    first, reference = np.asarray(first), np.asarray(reference)
    if labels.shape != first.shape or first.shape != reference.shape or set(labels) != {0, 1}:
        raise ValueError("Expected paired binary predictions with both classes.")
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap draws.")
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(labels == label) for label in (0, 1)]
    draws = {"auc": [], "macro_f1": []}
    for _ in range(repetitions):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in strata])
        y, a, b = labels[indices], first[indices], reference[indices]
        draws["auc"].append(roc_auc_score(y, a) - roc_auc_score(y, b))
        draws["macro_f1"].append(
            f1_score(y, a >= 0.5, average="macro", zero_division=0)
            - f1_score(y, b >= 0.5, average="macro", zero_division=0)
        )
    return {
        "unit": "participant", "stratification": "binary class", "seed": seed,
        "repetitions": repetitions, "interval": "conditional percentile 95%",
        "intervals": {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in draws.items()},
    }
