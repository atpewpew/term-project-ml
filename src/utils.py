"""Utility helpers for reproducibility, persistence, and text cleaning."""

from __future__ import annotations

import pickle
import random
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7E]")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and random for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: Path | str) -> None:
    """Create the directory (and parents) if it does not already exist."""

    Path(path).mkdir(parents=True, exist_ok=True)


def save_pickle(obj: object, path: Path | str) -> None:
    """Persist ``obj`` as a pickle file."""

    path = Path(path)
    ensure_directory(path.parent)
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: Path | str) -> object:
    """Load a pickle object from disk."""

    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def strip_html(text: str) -> str:
    """Remove HTML tags."""

    return HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """Remove URL patterns."""

    return URL_RE.sub(" ", text)


def strip_mentions(text: str) -> str:
    """Remove Twitter style mentions."""

    return MENTION_RE.sub(" ", text)


def strip_non_printable(text: str) -> str:
    """Remove non printable characters."""

    return NON_PRINTABLE_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Replace multiple whitespace with a single space."""

    return MULTISPACE_RE.sub(" ", text).strip()


def tokenize_text(text: str) -> list[str]:
    """Tokenize using a simple regex that keeps alphanumerics and apostrophes."""

    return TOKEN_RE.findall(text.lower())


def chunks(iterable: Iterable, chunk_size: int) -> Iterable[list]:
    """Yield successive chunks from an iterable."""

    chunk: list = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = [
    "set_global_seed",
    "ensure_directory",
    "save_pickle",
    "load_pickle",
    "strip_html",
    "strip_urls",
    "strip_mentions",
    "strip_non_printable",
    "collapse_whitespace",
    "tokenize_text",
    "chunks",
]
