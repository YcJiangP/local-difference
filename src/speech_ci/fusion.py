"""Binary and multiclass bounded logit fusion."""

import numpy as np
from scipy.special import expit


def logit(probabilities):
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def center(values):
    return values - np.mean(values, axis=-1, keepdims=True)


def softmax(values):
    values = np.asarray(values, dtype=float)
    exp_values = np.exp(values - values.max(axis=-1, keepdims=True))
    return exp_values / exp_values.sum(axis=-1, keepdims=True)


def probability_margins(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Expected finite participant-by-class probabilities.")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(-1), 1):
        raise ValueError("Invalid class probabilities.")
    classes = probabilities.shape[-1]
    return classes / (classes - 1) * center(np.log(np.clip(probabilities, 1e-12, 1)))


def from_margins(scores):
    classes = scores.shape[-1]
    return softmax((classes - 1) / classes * np.asarray(scores, dtype=float))


def binary_fusion(neural_probability, delta_logit, static_logit):
    neural = 2 * np.tanh(logit(neural_probability) / 4)
    static = np.tanh(static_logit)
    return {
        "N": np.asarray(neural_probability),
        "Delta": expit(delta_logit),
        "S": expit(static_logit),
        "N+Delta": expit(delta_logit + neural),
        "N+S": expit(neural + static),
        "Delta+S": expit(delta_logit + static),
        "Full": expit(delta_logit + neural + static),
    }


def multiclass_fusion(neural_probability, delta_scores, static_scores):
    neural = 2 * np.tanh(probability_margins(neural_probability) / 4)
    static = np.tanh(static_scores)
    return {
        "N": np.asarray(neural_probability),
        "Delta": from_margins(delta_scores),
        "S": from_margins(static_scores),
        "N+Delta": from_margins(delta_scores + neural),
        "N+S": from_margins(neural + static),
        "Delta+S": from_margins(delta_scores + static),
        "Full": from_margins(delta_scores + neural + static),
    }
