"""Abstract base class for custom classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseClassifier(ABC):
    """Common API for all classifiers used in the benchmark."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Train the classifier and return ``self`` for chaining."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilistic predictions when supported."""

        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return decision scores when supported."""

        raise NotImplementedError(f"{self.__class__.__name__} does not implement decision_function")

    def __repr__(self) -> str:  # pragma: no cover - convenience helper
        return f"{self.__class__.__name__}(name='{self.name}')"


__all__ = ["BaseClassifier"]
