"""Random forest ensemble built on the in-house decision tree.

Fast and sparse-friendly:
- Uses CSC/CSR optimised tree training (see decision_tree.py).
- Stratified bootstrap sampling per tree to balance classes.
- Progress reporting per tree with ETA.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from .. import config
from .base_model import BaseClassifier
from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier(BaseClassifier):
    """Bagging ensemble of custom decision trees."""

    def __init__(
        self,
        n_estimators: int = config.RANDOM_FOREST_N_ESTIMATORS,
        max_depth: Optional[int] = config.RANDOM_FOREST_MAX_DEPTH,
        min_samples_split: int = config.RANDOM_FOREST_MIN_SAMPLES_SPLIT,
        max_features_ratio: float | None = config.RANDOM_FOREST_MAX_FEATURES_RATIO,
        random_state: int = config.RANDOM_SEED,
        show_progress: bool = True,
    ) -> None:
        super().__init__(name="rf")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_ratio = max_features_ratio
        self.random_state = random_state
        self.show_progress = show_progress
        self.classes_: np.ndarray | None = None
        self.trees: list[DecisionTreeClassifier] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        X_train = X.tocsr() if sparse.isspmatrix(X) else sparse.csr_matrix(np.asarray(X))
        y_train = np.asarray(y, dtype=np.int64)
        classes = np.unique(y_train)
        if classes.size != 2:
            raise ValueError("RandomForestClassifier supports binary targets only")
        self.classes_ = classes

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X_train.shape
        # Determine number of features per split
        if self.max_features_ratio is None:
            max_features = int(max(1, np.sqrt(n_features)))
        else:
            mf = int(np.clip(self.max_features_ratio, 1e-3, 1.0) * n_features)
            max_features = max(1, mf)

        if self.show_progress:
            print(
                f"    [rf] Training: n_samples={n_samples:,}, n_features={n_features:,}, "
                f"n_estimators={self.n_estimators}, max_depth={self.max_depth}, max_features={max_features}",
                flush=True,
            )

        self.trees = []
        start = time.perf_counter()

        # Precompute class indices for balanced bootstrap
        idx_cls0 = np.flatnonzero(y_train == classes[0])
        idx_cls1 = np.flatnonzero(y_train == classes[1])
        half = n_samples // 2

        for i in range(self.n_estimators):
            # Stratified bootstrap: equal samples from each class (with replacement)
            b0 = rng.choice(idx_cls0, size=half, replace=True)
            b1 = rng.choice(idx_cls1, size=n_samples - half, replace=True)
            boot_idx = np.concatenate([b0, b1])

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X_train[boot_idx], y_train[boot_idx])
            self.trees.append(tree)

            if self.show_progress:
                elapsed = time.perf_counter() - start
                speed = (i + 1) / elapsed if elapsed > 0 else 0.0
                eta = (self.n_estimators - (i + 1)) / speed if speed > 0 else float("inf")
                print(
                    f"        built {i + 1}/{self.n_estimators} | elapsed {elapsed:5.1f}s | "
                    f"trees/s {speed:4.2f} | eta {eta:5.1f}s",
                    flush=True,
                )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest has not been fitted")
        X_csr = X.tocsr() if sparse.isspmatrix(X) else sparse.csr_matrix(np.asarray(X))
        # Majority vote. We must use the stored `self.classes_` to map the
        # result of bincount (0 or 1) back to the original class label.
        votes = np.column_stack([tree.predict(X_csr) for tree in self.trees])
        # bincount per row
        pred_indices = np.apply_along_axis(
            lambda r: np.argmax(np.bincount(r, minlength=self.classes_.size)),
            axis=1,
            arr=votes,
        )
        return self.classes_[pred_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Forest has not been fitted")
        X_csr = X.tocsr() if sparse.isspmatrix(X) else sparse.csr_matrix(np.asarray(X))
        votes = np.column_stack([tree.predict(X_csr) for tree in self.trees]).astype(float)
        proba_pos = votes.mean(axis=1)
        return np.column_stack([1.0 - proba_pos, proba_pos])


__all__ = ["RandomForestClassifier"]
