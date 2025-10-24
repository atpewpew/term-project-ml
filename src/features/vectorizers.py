"""Feature extraction pipelines built on scikit-learn vectorizers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from scipy import sparse

from .. import config
from ..utils import ensure_directory, load_pickle, save_pickle


def build_vectorizer(name: str) -> BaseVectorizer:
    """Build a vectorizer from a string identifier."""
    if name == "bow":
        return CountVectorizerWrapper(max_features=config.MAX_FEATURES_BOW_TFIDF)
    if name == "tfidf":
        return TfidfVectorizerWrapper(max_features=config.MAX_FEATURES_BOW_TFIDF)
    if name == "tfidf_bigram":
        return TfidfBigramVectorizerWrapper(max_features=config.MAX_FEATURES_TFIDF_BIGRAMS)
    raise ValueError(f"Unknown vectorizer '{name}'")


class BaseVectorizer(ABC):
    """Abstract base class for vectorizers."""

    name: str
    vectorizer: CountVectorizer | TfidfVectorizer

    def __init__(self, name: str, max_features: int | None) -> None:
        self.name = name
        self.max_features = max_features

    @abstractmethod
    def fit_transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        """Fit to data, then transform it."""

    @abstractmethod
    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        """Transform documents to document-term matrix."""


class CountVectorizerWrapper(BaseVectorizer):
    """Bag-of-words vectorizer."""

    def __init__(self, max_features: int | None = None) -> None:
        super().__init__(name="bow", max_features=max_features)
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english" if config.REMOVE_STOPWORDS else None,
        )

    def fit_transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(texts)


class TfidfVectorizerWrapper(BaseVectorizer):
    """TF-IDF vectorizer."""

    def __init__(self, max_features: int | None = None) -> None:
        super().__init__(name="tfidf", max_features=max_features)
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words="english" if config.REMOVE_STOPWORDS else None,
        )
        self.tfidf_transformer = TfidfTransformer(use_idf=True)

    def fit_transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        counts = self.count_vectorizer.fit_transform(texts)
        return self.tfidf_transformer.fit_transform(counts)

    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        counts = self.count_vectorizer.transform(texts)
        return self.tfidf_transformer.transform(counts)


class TfidfBigramVectorizerWrapper(BaseVectorizer):
    """TF-IDF bigram vectorizer."""

    def __init__(self, max_features: int | None = None) -> None:
        super().__init__(name="tfidf_bigram", max_features=max_features)
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            stop_words="english" if config.REMOVE_STOPWORDS else None,
        )

    def fit_transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: Iterable[str]) -> sparse.csr_matrix:
        return self.vectorizer.transform(texts)


__all__ = [
    "BaseVectorizer",
    "CountVectorizerWrapper",
    "TfidfVectorizerWrapper",
    "TfidfBigramVectorizerWrapper",
    "build_vectorizer",
]
