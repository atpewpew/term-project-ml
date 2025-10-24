"""Dataset loading utilities for the sentiment benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .preprocessing import clean_text_series


@dataclass(slots=True)
class DatasetSplit:
    """Container for standardized train/test data."""

    name: str
    train: pd.DataFrame
    test: pd.DataFrame


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {path}. Please place the file as described in the README."
        )


def _standardize(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    subset = df[[text_col, label_col]].copy()
    subset.columns = ["text", "label"]
    subset.dropna(subset=["text", "label"], inplace=True)
    subset["text"] = subset["text"].astype(str)
    subset["label"] = subset["label"].astype(int)
    return subset.reset_index(drop=True)


def _load_amazon() -> DatasetSplit:
    path = config.AMAZON_DATA
    _assert_exists(path)
    data = pd.read_csv(path)
    if "Sentiment" not in data.columns:
        raise ValueError("Amazon dataset must contain a 'Sentiment' column.")
    data.loc[data["Sentiment"] <= 3, "Sentiment"] = 0
    data.loc[data["Sentiment"] > 3, "Sentiment"] = 1
    standardized = _standardize(data, "Review", "Sentiment")
    train, test = train_test_split(
        standardized,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=standardized["label"],
    )
    train["clean_text"] = clean_text_series(train["text"], dataset="amazon")
    test["clean_text"] = clean_text_series(test["text"], dataset="amazon")
    return DatasetSplit(name="amazon", train=train, test=test)


def _load_imdb() -> DatasetSplit:
    path = config.IMDB_DATA
    _assert_exists(path)
    data = pd.read_csv(path)
    if {"review", "sentiment"}.difference(data.columns):
        raise ValueError("IMDB dataset must contain 'review' and 'sentiment'.")
    mapping = {"positive": 1, "negative": 0}
    data["sentiment"] = data["sentiment"].map(mapping)
    standardized = _standardize(data, "review", "sentiment")
    train, test = train_test_split(
        standardized,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
        stratify=standardized["label"],
    )
    train["clean_text"] = clean_text_series(train["text"], dataset="imdb")
    test["clean_text"] = clean_text_series(test["text"], dataset="imdb")
    return DatasetSplit(name="imdb", train=train, test=test)


def _load_twitter() -> DatasetSplit:
    train_path = config.TWITTER_TRAIN_DATA
    _assert_exists(train_path)
    train_data = pd.read_csv(train_path)
    if {"text", "label"}.difference(train_data.columns):
        raise ValueError("Twitter dataset must contain 'text' and 'label'.")

    if config.TWITTER_TEST_DATA.exists():
        test_data = pd.read_csv(config.TWITTER_TEST_DATA)
    else:
        train_data, test_data = train_test_split(
            train_data,
            test_size=0.2,
            random_state=config.RANDOM_SEED,
            stratify=train_data["label"],
        )

    train = _standardize(train_data, "text", "label")
    test = _standardize(test_data, "text", "label")
    train["clean_text"] = clean_text_series(train["text"], dataset="twitter")
    test["clean_text"] = clean_text_series(test["text"], dataset="twitter")
    return DatasetSplit(name="twitter", train=train, test=test)


LOADERS: dict[str, Callable[[], DatasetSplit]] = {
    "amazon": _load_amazon,
    "imdb": _load_imdb,
    "twitter": _load_twitter,
}


def load_dataset(name: str) -> DatasetSplit:
    """Load and preprocess the requested dataset."""

    name = name.lower()
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(LOADERS)}")
    dataset = LOADERS[name]()
    return dataset


__all__ = ["DatasetSplit", "load_dataset", "LOADERS"]
