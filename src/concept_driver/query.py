from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from concept_driver.data import (
    build_texts_for_embedding,
    extract_contexts,
    load_concepts,
    read_corpus,
)
from concept_driver.embeddings import (
    DEFAULT_MODEL,
    EmbeddingBackend,
    aggregate_term_embeddings,
    build_backend,
    resolve_model_name,
)


@dataclass
class QueryMatch:
    term: str
    score: float
    concept_set: str
    language: str


@dataclass
class QueryResult:
    query: str
    known_term: bool
    has_signal: bool
    concept_sets: list[str]
    contexts: list[str]
    neighbors: list[QueryMatch]


@dataclass
class QuerySession:
    concepts: pd.DataFrame
    corpus_text: str
    mode: str
    encoder: str
    model_name: str
    max_contexts: int
    context_window: int
    backend: EmbeddingBackend
    term_embeddings: np.ndarray
    term_to_contexts: dict[str, list[str]]
    embedding_task_type: str | None = None
    query_task_type: str | None = None

    def _query_contexts(self, term: str) -> list[str]:
        known_contexts = self.term_to_contexts.get(term)
        if known_contexts is not None:
            return known_contexts

        if self.mode == "context" and self.corpus_text:
            contexts = extract_contexts(
                self.corpus_text,
                [term],
                max_contexts=self.max_contexts,
                context_window=self.context_window,
            )[term]
            if contexts:
                return contexts

        return [term]

    def _embed_query(self, term: str) -> tuple[np.ndarray, list[str]]:
        contexts = self._query_contexts(term)
        embeddings = self.backend.transform(contexts, task_type=self.query_task_type)
        query_vector = embeddings.mean(axis=0, keepdims=True).astype(np.float32)
        query_vector = normalize(query_vector)
        return query_vector, contexts

    def query(self, term: str, top_k: int = 5) -> QueryResult:
        query_vector, contexts = self._embed_query(term)
        similarity = cosine_similarity(query_vector, self.term_embeddings)[0]
        known_mask = self.concepts["term"].astype(str).str.casefold() == term.casefold()
        known_term = bool(known_mask.any())
        concept_sets = sorted(self.concepts.loc[known_mask, "concept_set"].astype(str).unique().tolist())
        has_signal = bool(float(np.max(similarity)) > 0.0)

        order = np.argsort(-similarity)
        neighbors: list[QueryMatch] = []
        if has_signal:
            for idx in order:
                row = self.concepts.iloc[int(idx)]
                if known_term and str(row["term"]).casefold() == term.casefold():
                    continue
                neighbors.append(
                    QueryMatch(
                        term=str(row["term"]),
                        score=float(similarity[int(idx)]),
                        concept_set=str(row["concept_set"]),
                        language=str(row.get("language", "")),
                    )
                )
                if len(neighbors) >= top_k:
                    break

        return QueryResult(
            query=term,
            known_term=known_term,
            has_signal=has_signal,
            concept_sets=concept_sets,
            contexts=contexts,
            neighbors=neighbors,
        )


def build_query_session(
    concepts_path: str | None,
    corpus_path: str | None,
    mode: str,
    encoder: str,
    model_name: str = DEFAULT_MODEL,
    max_contexts: int = 25,
    context_window: int = 1,
    concepts_df: pd.DataFrame | None = None,
    embedding_task_type: str | None = None,
    query_task_type: str | None = None,
    output_dimensionality: int | None = None,
    gemini_api_key: str | None = None,
) -> QuerySession:
    if concepts_df is not None:
        concepts = concepts_df.copy()
    elif concepts_path is not None:
        concepts = load_concepts(concepts_path)
    else:
        raise ValueError("Either concepts_path or concepts_df must be provided.")
    corpus_text = read_corpus(corpus_path)
    all_texts, text_index = build_texts_for_embedding(
        concepts,
        mode=mode,
        corpus_text=corpus_text,
        max_contexts=max_contexts,
        context_window=context_window,
    )
    resolved_model_name = resolve_model_name(encoder, model_name)
    backend, all_embeddings = build_backend(
        all_texts,
        encoder=encoder,
        model_name=resolved_model_name,
        task_type=embedding_task_type,
        output_dimensionality=output_dimensionality,
        api_key=gemini_api_key,
    )
    term_embeddings, term_to_contexts = aggregate_term_embeddings(
        concepts,
        text_index,
        all_texts,
        all_embeddings,
    )
    return QuerySession(
        concepts=concepts,
        corpus_text=corpus_text,
        mode=mode,
        encoder=encoder,
        model_name=resolved_model_name,
        max_contexts=max_contexts,
        context_window=context_window,
        backend=backend,
        term_embeddings=term_embeddings,
        term_to_contexts=term_to_contexts,
        embedding_task_type=embedding_task_type,
        query_task_type=query_task_type,
    )
