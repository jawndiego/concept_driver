from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

try:
    from ripser import ripser  # type: ignore
except Exception:  # pragma: no cover
    ripser = None


def pca_project(X: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    n_components = min(n_components, X.shape[0], X.shape[1])
    if n_components < 1:
        raise ValueError("Not enough data for PCA")

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(X)
    return projected, pca.explained_variance_ratio_


def umap_project(
    X: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
) -> np.ndarray:
    if umap is None or X.shape[0] < 4:
        projected, _ = pca_project(X, n_components=2)
        if projected.shape[1] == 1:
            projected = np.hstack([projected, np.zeros((len(projected), 1), dtype=projected.dtype)])
        return projected[:, :2]

    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, max(2, X.shape[0] - 1)),
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def knn_edges(X: np.ndarray, k: int) -> list[tuple[int, int]]:
    if X.shape[0] <= 1:
        return []

    k = max(1, min(k, X.shape[0] - 1))
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    _, indices = neighbors.kneighbors(X)

    edges = set()
    for left, row in enumerate(indices):
        for right in row[1:]:
            edge = tuple(sorted((left, int(right))))
            edges.add(edge)

    return sorted(edges)


def persistent_h1_summary(X: np.ndarray) -> dict[str, float]:
    if ripser is None or X.shape[0] < 4:
        return {
            "h1_count": 0.0,
            "h1_max_persistence": 0.0,
            "h1_mean_persistence": 0.0,
        }

    diagrams = ripser(X, maxdim=1, metric="cosine")["dgms"]
    if len(diagrams) < 2 or len(diagrams[1]) == 0:
        return {
            "h1_count": 0.0,
            "h1_max_persistence": 0.0,
            "h1_mean_persistence": 0.0,
        }

    persistence = diagrams[1][:, 1] - diagrams[1][:, 0]
    finite = persistence[np.isfinite(persistence)]
    if len(finite) == 0:
        return {
            "h1_count": float(len(persistence)),
            "h1_max_persistence": 0.0,
            "h1_mean_persistence": 0.0,
        }

    return {
        "h1_count": float(len(persistence)),
        "h1_max_persistence": float(np.max(finite)),
        "h1_mean_persistence": float(np.mean(finite)),
    }


def nearest_neighbors_table(
    X: np.ndarray,
    labels: Sequence[str],
    top_k: int = 4,
) -> pd.DataFrame:
    similarity = cosine_similarity(X)
    rows = []
    for idx, label in enumerate(labels):
        order = np.argsort(-similarity[idx])
        neighbors = [
            (labels[other], float(similarity[idx, other]))
            for other in order[1 : top_k + 1]
        ]
        rows.append(
            {
                "label": label,
                "neighbors": ", ".join(
                    f"{name} ({score:.2f})" for name, score in neighbors
                ),
            }
        )

    return pd.DataFrame(rows)
