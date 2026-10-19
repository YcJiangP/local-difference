"""Class-balanced paired linear classifiers."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.preprocessing import StandardScaler

from .fusion import center, from_margins, softmax
from .io import load_npz, save_npz
from .models import class_weights


class PairedHead:
    """Binary or multiclass head, with an unpenalized intercept."""

    def __init__(self, c=1e-3, beta=1.0, augmented=True, classes=2):
        if c <= 0 or beta < 0 or classes not in (2, 3):
            raise ValueError("Invalid head configuration.")
        self.c = float(c)
        self.beta = float(beta)
        self.augmented = bool(augmented)
        self.classes = classes

    def objective(self, parameters, x0, x1, y, weights):
        if self.classes == 2:
            w, intercept = parameters[:-1], parameters[-1]
            z0 = x0 @ w + intercept
            residual = weights * (expit(z0) - y)
            loss = np.mean(weights * (np.logaddexp(0, z0) - y * z0))
            grad_w, grad_b = x0.T @ residual / len(y), residual.mean()
            if self.augmented:
                z1 = x1 @ w + intercept
                residual1 = weights * (expit(z1) - y)
                loss = 0.5 * (loss + np.mean(weights * (np.logaddexp(0, z1) - y * z1)))
                grad_w = 0.5 * (grad_w + x1.T @ residual1 / len(y))
                grad_b = 0.5 * (grad_b + residual1.mean())
                difference = x1 - x0
                drift = difference @ w
                loss += 0.5 * self.beta * np.mean(weights * drift ** 2)
                grad_w += self.beta * difference.T @ (weights * drift) / len(y)
            regularization = 1 / (self.c * len(y))
            loss += 0.5 * regularization * np.dot(w, w)
            grad_w += regularization * w
            return float(loss), np.r_[grad_w, grad_b]
        n, dimension = x0.shape
        classes = self.classes
        w = center(parameters[:dimension * classes].reshape(dimension, classes))
        intercept = center(parameters[dimension * classes:])
        one_hot = np.eye(classes)[y]
        scale = (classes - 1) / classes

        def cross_entropy(scores):
            logits = scale * scores
            probabilities = softmax(logits)
            value = np.mean(weights * (logsumexp(logits, axis=1) - logits[np.arange(n), y]))
            gradient = scale * weights[:, None] * (probabilities - one_hot) / n
            return value, gradient

        loss, grad0 = cross_entropy(x0 @ w + intercept)
        grad_w, grad_b = x0.T @ grad0, grad0.sum(0)
        if self.augmented:
            loss1, grad1 = cross_entropy(x1 @ w + intercept)
            loss = 0.5 * (loss + loss1)
            grad_w = 0.5 * (grad_w + x1.T @ grad1)
            grad_b = 0.5 * (grad_b + grad1.sum(0))
            difference = x1 - x0
            drift = difference @ w
            loss += 0.5 * self.beta * np.mean(weights * np.mean(drift ** 2, axis=1))
            grad_w += self.beta * difference.T @ (weights[:, None] * drift) / (n * classes)
        regularization = 1 / (self.c * n)
        loss += 0.5 * regularization * np.sum(w * w) / classes
        grad_w += regularization * w / classes
        return float(loss), np.r_[center(grad_w).ravel(), center(grad_b)]

    def fit(self, original, augmented, labels):
        x0, x1 = np.asarray(original, dtype=float), np.asarray(augmented, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if x0.ndim != 2 or x1.shape != x0.shape or len(labels) != len(x0):
            raise ValueError("Paired features and labels are misaligned.")
        if not np.isfinite(x0).all() or not np.isfinite(x1).all():
            raise ValueError("Non-finite descriptor values.")
        if set(labels) != set(range(self.classes)):
            raise ValueError("Fitting labels must contain every class.")
        scaler = StandardScaler().fit(x0)
        self.mean_, self.scale_ = scaler.mean_.copy(), scaler.scale_.copy()
        x0, x1 = scaler.transform(x0), scaler.transform(x1)
        weights = class_weights(labels, self.classes)[labels]
        count = (x0.shape[1] + 1) * (1 if self.classes == 2 else self.classes)
        result = minimize(
            lambda values: self.objective(values, x0, x1, labels, weights),
            np.zeros(count),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1500, "ftol": 1e-13, "gtol": 1e-8, "maxls": 40},
        )
        if not result.success:
            raise RuntimeError(f"Linear-head optimization did not converge: {result.message}")
        if self.classes == 2:
            self.coef_, self.intercept_ = result.x[:-1], np.asarray(result.x[-1])
        else:
            size = x0.shape[1] * self.classes
            self.coef_ = center(result.x[:size].reshape(-1, self.classes))
            self.intercept_ = center(result.x[size:])
        self.diagnostics = {
            "success": True, "iterations": int(result.nit), "objective": float(result.fun),
            "gradient_max": float(np.abs(result.jac).max()),
        }
        return self

    def decision_function(self, features):
        values = (np.asarray(features, dtype=float) - self.mean_) / self.scale_
        scores = values @ self.coef_ + self.intercept_
        return scores if self.classes == 2 else center(scores)

    def predict_proba(self, features):
        scores = self.decision_function(features)
        if self.classes == 2:
            positive = expit(scores)
            return np.column_stack((1 - positive, positive))
        return from_margins(scores)

    def save(self, path):
        save_npz(
            path, coef=self.coef_, intercept=self.intercept_,
            mean=self.mean_, scale=self.scale_, classes=np.asarray(self.classes),
            c=np.asarray(self.c), beta=np.asarray(self.beta), augmented=np.asarray(self.augmented),
        )

    @classmethod
    def load(cls, path):
        values = load_npz(path)
        head = cls(float(values["c"]), float(values["beta"]), bool(values["augmented"]), int(values["classes"]))
        head.coef_, head.intercept_ = values["coef"], values["intercept"]
        head.mean_, head.scale_ = values["mean"], values["scale"]
        return head
