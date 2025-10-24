"""Simple CART-style decision tree optimised for sparse text features.

Key optimisations:
- Uses CSC representation during training for fast column (feature) access.
- Computes split scores via index-set operations (no dense masks/materialisation).
- Splits on feature presence (value > 0) which fits BOW/TF-IDF inputs well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse

from .. import config
from .base_model import BaseClassifier


@dataclass(slots=True)
class _TreeNode:
    prediction: int
    feature: Optional[int] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier(BaseClassifier):
    """Binary decision tree that splits on feature presence (value > 0)."""

    def __init__(
        self,
        max_depth: Optional[int] = config.RANDOM_FOREST_MAX_DEPTH,
        min_samples_split: int = config.RANDOM_FOREST_MIN_SAMPLES_SPLIT,
        max_features: Optional[int] = None,
        random_state: Optional[int] = config.RANDOM_SEED,
    ) -> None:
        super().__init__(name="tree")
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.tree_: Optional[_TreeNode] = None
        self.n_features_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        # Keep CSR for row-wise prediction, build CSC for feature-wise splits.
        X_csr = X.tocsr() if sparse.isspmatrix(X) else sparse.csr_matrix(np.asarray(X))
        X_csc = X_csr.tocsc()
        y = np.asarray(y, dtype=np.int64)
        self.n_features_ = X_csr.shape[1]
        rng = np.random.default_rng(self.random_state)

        # Precompute class sums optionally if needed per node (we use on-the-fly sums).

        def gini_counts(n_pos: int, n_total: int) -> float:
            if n_total == 0:
                return 0.0
            p = n_pos / n_total
            return 1.0 - (p * p + (1.0 - p) * (1.0 - p))

        def majority(labels_idx: np.ndarray) -> int:
            # labels_idx is array of row indices
            s = int(y[labels_idx].sum())
            return 1 if s * 2 >= labels_idx.size else 0

        def sample_features() -> np.ndarray:
            n = self.n_features_
            if self.max_features is None or self.max_features >= n:
                return np.arange(n)
            return rng.choice(n, size=self.max_features, replace=False)

        def build(node_idx: np.ndarray, depth: int) -> _TreeNode:
            # node_idx must remain sorted for efficient intersections
            node_idx.sort(kind="mergesort")
            n_total = node_idx.size
            pred = majority(node_idx)

            # stopping conditions
            if (self.max_depth is not None and depth >= self.max_depth) or n_total < self.min_samples_split:
                return _TreeNode(prediction=pred)
            # pure node
            unique_vals = np.unique(y[node_idx])
            if unique_vals.size == 1:
                return _TreeNode(prediction=int(unique_vals[0]))

            # current impurity
            node_pos = int(y[node_idx].sum())
            node_gini = gini_counts(node_pos, n_total)

            best_feature = None
            best_left_idx = None
            best_right_idx = None
            best_score = np.inf

            feats = sample_features()
            indptr = X_csc.indptr
            indices = X_csc.indices

            for f in feats:
                start, end = indptr[f], indptr[f + 1]
                col_rows = indices[start:end]  # sorted row indices where feature present
                if col_rows.size == 0:
                    continue
                # intersection of sorted arrays
                right_idx = np.intersect1d(node_idx, col_rows, assume_unique=False)
                r_n = right_idx.size
                if r_n == 0 or r_n == n_total:
                    continue
                r_pos = int(y[right_idx].sum())
                l_n = n_total - r_n
                l_pos = node_pos - r_pos

                left_gini = gini_counts(l_pos, l_n)
                right_gini = gini_counts(r_pos, r_n)
                split_score = (l_n * left_gini + r_n * right_gini) / n_total

                if split_score + 1e-12 < best_score and split_score + 1e-12 < node_gini:
                    best_score = split_score
                    best_feature = f
                    # compute left indices as set difference (sorted outputs)
                    best_right_idx = right_idx
                    best_left_idx = np.setdiff1d(node_idx, right_idx, assume_unique=False)

            if best_feature is None or best_left_idx is None or best_right_idx is None:
                return _TreeNode(prediction=pred)

            left_node = build(best_left_idx, depth + 1)
            right_node = build(best_right_idx, depth + 1)
            return _TreeNode(prediction=pred, feature=best_feature, left=left_node, right=right_node)

        root_idx = np.arange(X_csr.shape[0])
        self.tree_ = build(root_idx, 0)
        # store CSR for prediction
        self._X_format_for_pred = "csr"
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None:
            raise RuntimeError("Tree has not been fitted")
        X_csr = X.tocsr() if sparse.isspmatrix(X) else sparse.csr_matrix(np.asarray(X))
        out = np.empty(X_csr.shape[0], dtype=int)

        for i in range(X_csr.shape[0]):
            out[i] = self._predict_row_csr(X_csr, i)
        return out

    def _predict_row_csr(self, X_csr: sparse.csr_matrix, row_idx: int) -> int:
        node = self.tree_
        indptr = X_csr.indptr
        indices = X_csr.indices
        while node and not node.is_leaf:
            f = node.feature  # type: ignore[assignment]
            row_start, row_end = indptr[row_idx], indptr[row_idx + 1]
            row_cols = indices[row_start:row_end]
            # binary search for presence
            present = np.searchsorted(row_cols, f)  # type: ignore[arg-type]
            go_right = present < row_cols.size and row_cols[present] == f
            node = node.right if go_right else node.left
        return node.prediction if node else 0


__all__ = ["DecisionTreeClassifier"]
