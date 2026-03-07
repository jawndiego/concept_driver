from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from concept_driver.data import build_texts_for_embedding, load_concepts, read_corpus
from concept_driver.embeddings import DEFAULT_MODEL, aggregate_term_embeddings, encode_texts
from concept_driver.reporting import make_index_page, render_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build interactive concept-shape reports from term or context embeddings."
    )
    parser.add_argument("--concepts", required=True, help="CSV containing concept_set and term columns.")
    parser.add_argument("--corpus", help="Plain-text corpus file. Required for --mode context.")
    parser.add_argument("--out", required=True, help="Output directory for HTML reports.")
    parser.add_argument("--mode", choices=["term", "context"], default="term")
    parser.add_argument(
        "--encoder",
        choices=["sentence-transformer", "tfidf"],
        default="tfidf",
        help="Embedding backend. TF-IDF works offline; sentence-transformer is higher quality when installed.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    parser.add_argument("--max-contexts", type=int, default=25, help="Max contexts to keep per term in context mode.")
    parser.add_argument("--context-window", type=int, default=1, help="Sentence window size around each match.")
    parser.add_argument("--knn", type=int, default=3, help="Neighbors in the kNN graph.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args(argv)


def build_reports(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = load_concepts(args.concepts)
    corpus_text = read_corpus(args.corpus)
    all_texts, text_index = build_texts_for_embedding(
        concepts,
        mode=args.mode,
        corpus_text=corpus_text,
        max_contexts=args.max_contexts,
        context_window=args.context_window,
    )

    all_embeddings = encode_texts(all_texts, encoder=args.encoder, model_name=args.model)
    term_embeddings, term_to_contexts = aggregate_term_embeddings(
        concepts,
        text_index,
        all_texts,
        all_embeddings,
    )

    concepts = concepts.copy()
    concepts["_row_idx"] = np.arange(len(concepts))

    index_entries: list[tuple[str, str, int]] = []
    manifest_rows: list[dict[str, float | str]] = []
    run_info = {
        "mode": args.mode,
        "encoder": args.encoder,
        "model": args.model if args.encoder == "sentence-transformer" else "tfidf",
    }

    for group_name, group_df in concepts.groupby("concept_set", sort=True):
        indices = group_df["_row_idx"].to_numpy()
        group_matrix = term_embeddings[indices]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(group_name)).strip("_") or "group"
        filename = f"{safe_name}.html"
        contexts_subset = {
            term: term_to_contexts[term]
            for term in group_df["term"].astype(str).tolist()
        }

        metrics = render_report(
            out_dir / filename,
            str(group_name),
            group_df,
            group_matrix,
            contexts_subset,
            run_info=run_info,
            knn=args.knn,
            random_state=args.random_state,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
        )
        index_entries.append((str(group_name), filename, len(group_df)))
        manifest_rows.append({"concept_set": str(group_name), **metrics, "file": filename})

    make_index_page(out_dir, index_entries)
    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = build_reports(args)
    print(f"Wrote reports to {out_dir}")
    print(f"Open: {out_dir / 'index.html'}")
    return 0
