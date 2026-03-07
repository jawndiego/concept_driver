from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

REQUIRED_COLUMNS = {"concept_set", "term"}
OPTIONAL_COLUMNS = {"label", "language", "order", "color"}


def load_concepts(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["label"] = df["label"].fillna(df["term"])
    df["language"] = df["language"].fillna("")
    return df


def read_corpus(path: Optional[str | Path]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


def sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def extract_contexts(
    corpus_text: str,
    terms: Sequence[str],
    max_contexts: int = 25,
    context_window: int = 1,
) -> dict[str, list[str]]:
    sentences = sentence_split(corpus_text)
    contexts = {term: [] for term in terms}
    if not sentences:
        return contexts

    patterns = {
        term: re.compile(rf"(?i)(?<!\w){re.escape(term)}(?!\w)")
        for term in terms
    }

    for idx, sentence in enumerate(sentences):
        for term, pattern in patterns.items():
            if len(contexts[term]) >= max_contexts:
                continue
            if not pattern.search(sentence):
                continue

            start = max(0, idx - context_window)
            end = min(len(sentences), idx + context_window + 1)
            window_text = " ".join(sentences[start:end])
            contexts[term].append(window_text)

    return contexts


def build_texts_for_embedding(
    df: pd.DataFrame,
    mode: str,
    corpus_text: str,
    max_contexts: int,
    context_window: int,
) -> tuple[list[str], dict[str, list[str]]]:
    terms = df["term"].astype(str).tolist()
    if mode == "term":
        return terms, {term: [term] for term in terms}

    if not corpus_text:
        raise ValueError("--corpus is required when --mode context")

    contexts = extract_contexts(
        corpus_text,
        terms,
        max_contexts=max_contexts,
        context_window=context_window,
    )

    texts: list[str] = []
    text_index: dict[str, list[str]] = {}
    for term in terms:
        term_contexts = contexts.get(term, [])
        if not term_contexts:
            term_contexts = [term]
        text_index[term] = term_contexts
        texts.extend(term_contexts)

    return texts, text_index


def corpus_terms(
    corpus_text: str,
    language: str = "en",
    min_freq: int = 2,
    max_terms: int = 250,
) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{1,}", corpus_text.lower())
    counts = Counter(tokens)

    stopwords = ENGLISH_STOP_WORDS if language == "en" else set()
    filtered = [
        (term, freq)
        for term, freq in counts.items()
        if freq >= min_freq and term not in stopwords
    ]
    filtered.sort(key=lambda item: (-item[1], item[0]))
    return [term for term, _freq in filtered[:max_terms]]


def build_concepts_from_corpus(
    corpus_text: str,
    concept_set: str = "corpus_vocab",
    language: str = "en",
    min_freq: int = 2,
    max_terms: int = 250,
) -> pd.DataFrame:
    terms = corpus_terms(
        corpus_text,
        language=language,
        min_freq=min_freq,
        max_terms=max_terms,
    )
    return pd.DataFrame(
        {
            "concept_set": [concept_set] * len(terms),
            "term": terms,
            "label": terms,
            "language": [language] * len(terms),
            "order": list(range(1, len(terms) + 1)),
        }
    )
