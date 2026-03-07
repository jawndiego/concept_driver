from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

DEFAULT_MODEL = "all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None


class EmbeddingBackend:
    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class SentenceTransformerBackend(EmbeddingBackend):
    model_name: str
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self) -> object:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install the analysis extras or use --encoder tfidf."
            )
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


@dataclass
class TfidfBackend(EmbeddingBackend):
    _vectorizer: object | None = field(default=None, init=False, repr=False)

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        matrix = self._vectorizer.fit_transform(texts)
        dense = matrix.toarray().astype(np.float32)
        return normalize(dense)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("TF-IDF backend has not been fitted yet.")
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        return normalize(dense)


def build_backend(
    texts: Sequence[str],
    encoder: str,
    model_name: str = DEFAULT_MODEL,
) -> tuple[EmbeddingBackend, np.ndarray]:
    if encoder == "sentence-transformer":
        backend: EmbeddingBackend = SentenceTransformerBackend(model_name=model_name)
    else:
        backend = TfidfBackend()
    return backend, backend.fit_transform(texts)


def encode_texts(
    texts: Sequence[str],
    encoder: str,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    _, embeddings = build_backend(texts, encoder=encoder, model_name=model_name)
    return embeddings


def encode_texts_sentence_transformer(
    texts: Sequence[str],
    model_name: str,
) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install the analysis extras or use --encoder tfidf."
        )

    model = SentenceTransformer(model_name)
    embeddings = model.encode(list(texts), show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def encode_texts_tfidf(texts: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    dense = matrix.toarray().astype(np.float32)
    return normalize(dense)


def aggregate_term_embeddings(
    df: pd.DataFrame,
    text_index: dict[str, list[str]],
    all_texts: Sequence[str],
    all_embeddings: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    positions: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(all_texts):
        positions[text].append(idx)

    used_positions = {text: 0 for text in positions}
    term_embeddings = []
    term_to_contexts: dict[str, list[str]] = {}

    for term in df["term"].astype(str).tolist():
        term_contexts = text_index[term]
        indices: list[int] = []
        for context in term_contexts:
            location_list = positions[context]
            indices.append(location_list[used_positions[context]])
            used_positions[context] += 1

        embedding = all_embeddings[indices].mean(axis=0)
        term_embeddings.append(embedding)
        term_to_contexts[term] = term_contexts

    term_matrix = np.vstack(term_embeddings).astype(np.float32)
    return normalize(term_matrix), term_to_contexts
