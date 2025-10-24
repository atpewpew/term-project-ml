"""Evaluation utilities for classifier benchmarking."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import ensure_directory


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return the standard classification metrics."""

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return metrics


def save_metrics_csv(metrics: Dict[str, float], path: Path | str) -> None:
    """Persist metrics dictionary as a single-row CSV."""

    ensure_directory(Path(path).parent)
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def confusion_matrix_to_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(matrix, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path | str) -> None:
    df = confusion_matrix_to_df(y_true, y_pred)
    ensure_directory(Path(path).parent)
    df.to_csv(path, index=True)


def plot_metric_bar(metrics: pd.DataFrame, metric: str, title: str, path: Path | str) -> None:
    """Create a bar plot for a chosen metric across experiments."""

    ensure_directory(Path(path).parent)
    metrics.plot(kind="bar", x="configuration", y=metric, figsize=(10, 6))
    plt.title(title)
    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


__all__ = [
    "compute_metrics",
    "save_metrics_csv",
    "confusion_matrix_to_df",
    "save_confusion_matrix",
    "plot_metric_bar",
]
