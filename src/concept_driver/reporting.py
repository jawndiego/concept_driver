from __future__ import annotations

import html
import json
import textwrap
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

from concept_driver.geometry import (
    knn_edges,
    nearest_neighbors_table,
    pca_project,
    persistent_h1_summary,
    umap_project,
)


def make_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_field: str = "language",
) -> go.Figure:
    color_arg = None
    if color_field in df.columns and df[color_field].astype(str).str.len().sum() > 0:
        color_arg = color_field

    fig = px.scatter(
        df,
        x=x,
        y=y,
        text="label",
        color=color_arg,
        hover_data=[column for column in ["term", "language", "order"] if column in df.columns],
        title=title,
    )
    fig.update_traces(textposition="top center", marker={"size": 10, "opacity": 0.85})
    fig.update_layout(height=520)
    return fig


def add_edges(
    fig: go.Figure,
    coords: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> go.Figure:
    if coords.shape[1] < 2:
        return fig

    for left, right in edges:
        fig.add_trace(
            go.Scatter(
                x=[coords[left, 0], coords[right, 0]],
                y=[coords[left, 1], coords[right, 1]],
                mode="lines",
                line={"width": 1},
                hoverinfo="skip",
                showlegend=False,
                opacity=0.35,
            )
        )

    return fig


def make_heatmap(similarity: np.ndarray, labels: Sequence[str], title: str) -> go.Figure:
    fig = px.imshow(
        similarity,
        x=labels,
        y=labels,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title=title,
    )
    fig.update_layout(height=650)
    return fig


def make_context_html(term_to_contexts: dict[str, list[str]], max_show: int = 5) -> str:
    blocks = ["<h2>Example contexts</h2>"]
    for term, contexts in term_to_contexts.items():
        blocks.append(f"<h3>{html.escape(term)}</h3>")
        blocks.append("<ul>")
        for context in contexts[:max_show]:
            snippet = textwrap.shorten(context, width=220, placeholder="...")
            blocks.append(f"<li>{html.escape(snippet)}</li>")
        blocks.append("</ul>")

    return "\n".join(blocks)


def make_metrics_table(metrics: Mapping[str, float]) -> str:
    items = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{value:.4f}</td></tr>"
        for key, value in metrics.items()
    )
    return f"<table class='metrics'>{items}</table>"


def make_index_page(out_dir: Path, entries: list[tuple[str, str, int]]) -> None:
    links = "\n".join(
        f"<li><a href='{html.escape(filename)}'>{html.escape(name)}</a> ({count} terms)</li>"
        for name, filename, count in entries
    )

    html_text = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Concept Driver</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem auto; max-width: 900px; line-height: 1.45; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <h1>Concept Driver</h1>
  <p>Select a concept-set report.</p>
  <ul>{links}</ul>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def render_report(
    out_path: Path,
    group_name: str,
    group_df: pd.DataFrame,
    X: np.ndarray,
    term_to_contexts: dict[str, list[str]],
    run_info: Mapping[str, str],
    knn: int,
    random_state: int,
    umap_neighbors: int,
    umap_min_dist: float,
) -> dict[str, float]:
    labels = group_df["label"].astype(str).tolist()
    pca_coords, explained = pca_project(X, n_components=min(3, X.shape[0], X.shape[1]))
    pca_2d = pca_coords[:, :2]
    if pca_2d.shape[1] == 1:
        pca_2d = np.hstack([pca_2d, np.zeros((len(pca_2d), 1), dtype=pca_2d.dtype)])

    umap_coords = umap_project(
        X,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
    )
    similarity = cosine_similarity(X)
    edges = knn_edges(X, k=knn)
    neighbors_df = nearest_neighbors_table(X, labels)
    h1_metrics = persistent_h1_summary(X)

    metrics = {
        "n_terms": float(len(group_df)),
        "pca_pc1": float(explained[0]) if len(explained) > 0 else 0.0,
        "pca_pc2": float(explained[1]) if len(explained) > 1 else 0.0,
        "pca_pc3": float(explained[2]) if len(explained) > 2 else 0.0,
        "pca_pc1_plus_pc2": (
            float(explained[:2].sum())
            if len(explained) > 1
            else float(explained[0]) if len(explained) else 0.0
        ),
        **h1_metrics,
    }

    plot_df = group_df.copy()
    plot_df["pca1"] = pca_2d[:, 0]
    plot_df["pca2"] = pca_2d[:, 1]
    plot_df["umap1"] = umap_coords[:, 0]
    plot_df["umap2"] = umap_coords[:, 1]

    pca_figure = add_edges(make_scatter(plot_df, "pca1", "pca2", f"{group_name}: PCA"), pca_2d, edges)
    umap_figure = add_edges(
        make_scatter(plot_df, "umap1", "umap2", f"{group_name}: UMAP / fallback"),
        umap_coords,
        edges,
    )
    heatmap = make_heatmap(similarity, labels, f"{group_name}: cosine similarity")

    html_text = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>{html.escape(group_name)} - Concept Driver</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; line-height: 1.45; }}
    h1, h2 {{ margin-top: 1.2rem; }}
    .note {{ background: #f6f8fa; border-left: 4px solid #999; padding: 0.8rem 1rem; margin: 1rem 0; }}
    .metrics th {{ text-align: left; padding-right: 1rem; }}
    .metrics td, .metrics th {{ padding: 0.25rem 0.5rem; border-bottom: 1px solid #eee; }}
    .metrics {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
    table.nn {{ border-collapse: collapse; width: 100%; }}
    table.nn th, table.nn td {{ border-bottom: 1px solid #eee; padding: 0.4rem 0.5rem; text-align: left; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.3rem; }}
  </style>
</head>
<body>
  <p><a href='index.html'>Back to index</a></p>
  <h1>{html.escape(group_name)}</h1>
  <div class='note'>
    <strong>Run info:</strong> <code>{html.escape(json.dumps(dict(run_info)))}</code><br>
    Read PCA first. Treat UMAP as an exploratory view, not ground truth. If a pattern survives in the scatter and the heatmap, it is more likely to be meaningful.
  </div>

  <h2>Shape summary</h2>
  {make_metrics_table(metrics)}

  <h2>PCA view</h2>
  {pca_figure.to_html(full_html=False, include_plotlyjs='cdn')}

  <h2>UMAP / fallback view</h2>
  {umap_figure.to_html(full_html=False, include_plotlyjs=False)}

  <h2>Cosine similarity heatmap</h2>
  {heatmap.to_html(full_html=False, include_plotlyjs=False)}

  <h2>Nearest neighbors</h2>
  {neighbors_df.to_html(index=False, classes='nn', border=0, escape=False)}

  {make_context_html(term_to_contexts)}
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")
    return metrics
