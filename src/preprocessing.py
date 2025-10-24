"""Text preprocessing helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.feature_extraction import text as sk_text

from . import config
from .utils import (
    collapse_whitespace,
    strip_html,
    strip_mentions,
    strip_non_printable,
    strip_urls,
    tokenize_text,
)

try:
    from nltk.stem import SnowballStemmer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    SnowballStemmer = None  # type: ignore


ENGLISH_STOPWORDS = set(sk_text.ENGLISH_STOP_WORDS)
_STEMMER = SnowballStemmer("english") if SnowballStemmer else None


def _apply_optional_filters(tokens: Iterable[str]) -> list[str]:
    """Apply optional stopword removal and stemming configured in ``config``."""

    processed: list[str] = []
    for token in tokens:
        if config.REMOVE_STOPWORDS and token in ENGLISH_STOPWORDS:
            continue
        if config.APPLY_STEMMING and _STEMMER:
            token = _STEMMER.stem(token)
        processed.append(token)
    return processed


def clean_text(text: str, dataset: str | None = None) -> str:
    """Clean a single document and return the normalized string."""

    text = text.lower()
    text = strip_html(text)
    if dataset == "twitter":
        text = strip_mentions(text)
    text = strip_urls(text)
    text = strip_non_printable(text)
    tokens = tokenize_text(text)
    tokens = _apply_optional_filters(tokens)
    cleaned = collapse_whitespace(" ".join(tokens))
    return cleaned


def clean_text_series(series: pd.Series, dataset: str | None = None) -> pd.Series:
    """Vectorized helper to clean an entire pandas Series."""

    return series.astype(str).apply(lambda txt: clean_text(txt, dataset=dataset))


__all__ = ["clean_text", "clean_text_series"]
