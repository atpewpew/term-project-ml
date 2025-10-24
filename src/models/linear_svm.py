"""Linear SVM in the primal using stochastic gradient descent."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .. import config
from .base_model import BaseClassifier


class LinearSVMClassifier(BaseClassifier):
    """Primal linear SVM optimized with mini-batch SGD."""

    def __init__(
        self,
        lambda_param: float = config.SVM_LAMBDA,
        epochs: int = config.SVM_EPOCHS,
        batch_size: int = config.SVM_BATCH_SIZE,
        learning_rate: float | None = None,
    ) -> None:
        super().__init__(name="svm")
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.batch_size = batch_size
        # Use a sensible default if learning rate is not provided. This heuristic
        # is based on the optimal learning rate being proportional to 1/sqrt(lambda).
        self.learning_rate = learning_rate or 1.0 / np.sqrt(self.lambda_param)
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMClassifier":
        if sparse.isspmatrix(X):
            X = X.tocsr()
        else:
            X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] != 2:
            raise ValueError("LinearSVMClassifier supports binary classification only")
        y_signed = np.where(y == self.classes_[1], 1.0, -1.0)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0
        rng = np.random.default_rng(config.RANDOM_SEED)

        for epoch in range(self.epochs):
            # Simple learning rate decay
            lr = self.learning_rate / (1 + epoch)
            indices = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_X = X[batch_idx]
                batch_y = y_signed[batch_idx]

                decision = _safe_linear(batch_X, self.weights, self.bias)
                margin = batch_y * decision
                active_mask = margin < 1.0

                grad_w_reg = self.lambda_param * self.weights

                if np.any(active_mask):
                    active_y = batch_y[active_mask]
                    active_X = batch_X[active_mask]
                    # N.B. the gradient is the sum over misclassified samples,
                    # scaled by 1/N.
                    grad_w_hinge = -active_X.T @ active_y / batch_y.size
                    grad_b_hinge = -active_y.sum() / batch_y.size
                else:
                    grad_w_hinge = 0.0
                    grad_b_hinge = 0.0

                grad_w = grad_w_reg + grad_w_hinge
                grad_b = grad_b_hinge

                self.weights -= lr * grad_w
                self.bias -= lr * grad_b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")
        return _safe_linear(X, self.weights, self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        decision = self.decision_function(X)
        labels = np.where(decision >= 0, self.classes_[1], self.classes_[0])  # type: ignore[index]
        return labels


def _safe_linear(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    if sparse.isspmatrix(X):
        result = X @ weights
        return np.asarray(result).ravel() + bias
    return X @ weights + bias


__all__ = ["LinearSVMClassifier"]
