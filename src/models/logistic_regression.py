"""Binary logistic regression with mini-batch gradient descent."""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from .. import config
from .base_model import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """Simple binary logistic regression trained using SGD."""

    def __init__(
        self,
        learning_rate: float = config.LOGISTIC_REG_LR,
        epochs: int = config.LOGISTIC_REG_EPOCHS,
        batch_size: int = config.LOGISTIC_REG_BATCH_SIZE,
        l2_reg: float = config.LOGISTIC_REG_ALPHA,
    ) -> None:
        super().__init__(name="logreg")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        if sparse.isspmatrix(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)

        y_train = np.asarray(y, dtype=np.float64)
        self.classes_ = np.unique(y_train)
        if self.classes_.shape[0] != 2:
            raise ValueError("LogisticRegressionClassifier supports binary targets only")
        # Internally, model uses {0, 1} for labels
        y_internal = (y_train == self.classes_[1]).astype(np.float64)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        rng = np.random.default_rng(config.RANDOM_SEED)

        for epoch in range(self.epochs):
            lr = self.learning_rate * (0.95**epoch)
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_X = X[batch_idx]
                batch_y = y_internal[batch_idx]

                linear = _safe_linear(batch_X, self.weights, self.bias)
                predictions = 1.0 / (1.0 + np.exp(-linear))
                errors = predictions - batch_y

                grad_w = batch_X.T @ errors / batch_y.size
                grad_w = np.asarray(grad_w).ravel()
                grad_w += self.l2_reg * self.weights
                grad_b = errors.mean()

                self.weights -= lr * grad_w
                self.bias -= lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")
        linear = _safe_linear(X, self.weights, self.bias)
        proba_class1 = 1.0 / (1.0 + np.exp(-linear))
        proba = np.column_stack([1 - proba_class1, proba_class1])
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted")
        proba = self.predict_proba(X)
        class_indices = (proba[:, 1] >= 0.5).astype(int)
        return self.classes_[class_indices]


def _safe_linear(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Compute X @ weights + bias for dense or sparse matrices."""

    if sparse.isspmatrix(X):
        result = X @ weights
        return np.asarray(result).ravel() + bias
    return X @ weights + bias


__all__ = ["LogisticRegressionClassifier"]
