"""Multinomial Naive Bayes implemented from scratch."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .base_model import BaseClassifier


class MultinomialNaiveBayes(BaseClassifier):
    """Classic multinomial Naive Bayes classifier for count data."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(name="nb")
        if alpha <= 0:
            raise ValueError("alpha must be positive for Laplace smoothing")
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialNaiveBayes":
        y = np.asarray(y, dtype=np.int64)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")

        self.classes_, counts = np.unique(y, return_counts=True)
        n_classes = self.classes_.shape[0]
        n_features = X.shape[1]

        smoothed_fc = np.zeros((n_classes, n_features), dtype=np.float64)
        for idx, cls in enumerate(self.classes_):
            cls_rows = y == cls
            X_cls = X[cls_rows]
            class_feature_sum = X_cls.sum(axis=0)
            if sparse.isspmatrix(class_feature_sum):
                class_feature_sum = np.asarray(class_feature_sum).ravel()
            else:
                class_feature_sum = np.asarray(class_feature_sum).ravel()
            smoothed_fc[idx] = class_feature_sum + self.alpha

        feature_sum = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(feature_sum)
        self.class_log_prior_ = np.log(counts) - np.log(counts.sum())
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        if self.feature_log_prob_ is None or self.class_log_prior_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        if sparse.isspmatrix(X):
            return X @ self.feature_log_prob_.T + self.class_log_prior_
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        indices = np.argmax(jll, axis=1)
        return self.classes_[indices]  # type: ignore[index]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        jll = self._joint_log_likelihood(X)
        log_prob = jll - jll.max(axis=1, keepdims=True)
        prob = np.exp(log_prob)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob


__all__ = ["MultinomialNaiveBayes"]
