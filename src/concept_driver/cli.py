from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import webbrowser
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from concept_driver.data import (
    build_concepts_from_corpus,
    build_texts_for_embedding,
    load_concepts,
    read_corpus,
)
from concept_driver.embeddings import DEFAULT_MODEL, aggregate_term_embeddings, build_backend, encode_texts
from concept_driver.llm_concepts import generate_concepts_from_llm
from concept_driver.query import QueryResult, build_query_session
from concept_driver.remote_llm import RemoteLLMClient
from concept_driver.reporting import make_index_page, render_report


KNOWN_COMMANDS = {"report", "sample", "doctor", "tui"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def normalize_argv(argv: Sequence[str] | None = None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return args

    if args[0].startswith("-"):
        return ["report", *args]

    return args


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concepts", help="CSV containing concept_set and term columns.")
    parser.add_argument("--corpus", help="Plain-text corpus file. Required for --mode context or --auto-concepts.")
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
    parser.add_argument("--open", action="store_true", help="Open the generated index.html in your default browser.")
    parser.add_argument("--auto-concepts", action="store_true", help="Build a concept list directly from the corpus.")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum term frequency for --auto-concepts.")
    parser.add_argument("--max-terms", type=int, default=250, help="Maximum number of auto-generated corpus terms.")
    parser.add_argument("--language", default="en", help="Language tag for generated terms.")
    parser.add_argument("--concept-set-name", default="corpus_vocab", help="Concept set name used by --auto-concepts.")


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-base-url",
        help="OpenAI-compatible base URL, for example https://conceptdriver-production.up.railway.app/v1",
    )
    parser.add_argument("--llm-api-key", help="Bearer token for the remote LLM endpoint.")
    parser.add_argument("--llm-model", help="Model name sent to the remote endpoint. Defaults to the first /models entry.")
    parser.add_argument(
        "--llm-system",
        default=(
            "You are a concise concept analysis assistant. "
            "Define words clearly, mention related concepts, and avoid extra fluff."
        ),
        help="System prompt used for remote LLM requests.",
    )
    parser.add_argument("--llm-timeout", type=float, default=120.0, help="Timeout in seconds for remote LLM calls.")


def add_llm_report_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--llm-query", help="Generate a concept set from the remote LLM for this query.")
    parser.add_argument("--llm-max-terms", type=int, default=24, help="Maximum number of terms to request from the remote LLM.")
    parser.add_argument("--llm-concept-set-name", help="Optional report concept-set name for --llm-query.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and inspect concept-shape reports from term or context embeddings."
    )
    subparsers = parser.add_subparsers(dest="command")

    report_parser = subparsers.add_parser(
        "report",
        help="Build reports from your own concept CSV and optional corpus.",
    )
    add_report_arguments(report_parser)
    add_llm_arguments(report_parser)
    add_llm_report_arguments(report_parser)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Run the bundled example dataset with sensible defaults.",
    )
    sample_parser.add_argument(
        "--out",
        default=str(repo_root() / "output" / "sample-report"),
        help="Output directory for the sample HTML reports.",
    )
    sample_parser.add_argument("--mode", choices=["term", "context"], default="context")
    sample_parser.add_argument(
        "--encoder",
        choices=["sentence-transformer", "tfidf"],
        default="tfidf",
        help="Embedding backend for the sample run.",
    )
    sample_parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    sample_parser.add_argument("--max-contexts", type=int, default=25)
    sample_parser.add_argument("--context-window", type=int, default=1)
    sample_parser.add_argument("--knn", type=int, default=3)
    sample_parser.add_argument("--random-state", type=int, default=42)
    sample_parser.add_argument("--umap-neighbors", type=int, default=5)
    sample_parser.add_argument("--umap-min-dist", type=float, default=0.1)
    sample_parser.add_argument("--open", action="store_true", help="Open the generated sample report in your browser.")

    subparsers.add_parser(
        "doctor",
        help="Show available backends and whether optional dependencies are installed.",
    )

    tui_parser = subparsers.add_parser(
        "tui",
        help="Open an interactive terminal query loop.",
    )
    tui_parser.add_argument("--concepts", help="CSV containing concept_set and term columns.")
    tui_parser.add_argument("--corpus", help="Plain-text corpus file.")
    tui_parser.add_argument("--sample", action="store_true", help="Use the bundled toy example data.")
    tui_parser.add_argument("--mode", choices=["term", "context"], default="context")
    tui_parser.add_argument(
        "--encoder",
        choices=["sentence-transformer", "tfidf"],
        default="tfidf",
        help="Embedding backend for interactive queries.",
    )
    tui_parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name.")
    tui_parser.add_argument("--max-contexts", type=int, default=25)
    tui_parser.add_argument("--context-window", type=int, default=1)
    tui_parser.add_argument("--top-k", type=int, default=5, help="How many neighbors to show for each query.")
    tui_parser.add_argument("--auto-concepts", action="store_true", help="Build a concept list directly from the corpus.")
    tui_parser.add_argument("--min-freq", type=int, default=2, help="Minimum term frequency for --auto-concepts.")
    tui_parser.add_argument("--max-terms", type=int, default=250, help="Maximum number of auto-generated corpus terms.")
    tui_parser.add_argument("--language", default="en", help="Language tag for generated terms.")
    tui_parser.add_argument("--concept-set-name", default="corpus_vocab", help="Concept set name used by --auto-concepts.")
    add_llm_arguments(tui_parser)

    args = parser.parse_args(normalize_argv(argv))
    if args.command is None:
        parser.print_help()
        parser.exit(1)

    if args.command == "sample":
        args.concepts = str(repo_root() / "examples" / "example_concepts.csv")
        args.corpus = str(repo_root() / "examples" / "example_corpus.txt")

    return args


def resolve_concepts(args: argparse.Namespace) -> pd.DataFrame:
    if getattr(args, "command", None) == "sample":
        return load_concepts(args.concepts)

    if getattr(args, "sample", False):
        args.concepts = str(repo_root() / "examples" / "example_concepts.csv")
        args.corpus = str(repo_root() / "examples" / "example_corpus.txt")
        return load_concepts(args.concepts)

    if getattr(args, "auto_concepts", False):
        if not args.corpus:
            raise ValueError("--corpus is required when --auto-concepts is enabled.")
        corpus_text = read_corpus(args.corpus)
        concepts = build_concepts_from_corpus(
            corpus_text,
            concept_set=args.concept_set_name,
            language=args.language,
            min_freq=args.min_freq,
            max_terms=args.max_terms,
        )
        if concepts.empty:
            raise ValueError("Auto-generated concept list is empty. Lower --min-freq or use a larger corpus.")
        return concepts

    if getattr(args, "llm_query", None):
        if args.concepts:
            raise ValueError("Use either --concepts or --llm-query, not both.")
        if getattr(args, "auto_concepts", False):
            raise ValueError("Use either --auto-concepts or --llm-query, not both.")

        llm_client = build_llm_client(args)
        if llm_client is None:
            raise ValueError("--llm-base-url is required when --llm-query is used.")

        concepts = generate_concepts_from_llm(
            llm_client,
            query=args.llm_query,
            concept_set_name=args.llm_concept_set_name,
            language=args.language,
            max_terms=args.llm_max_terms,
        )
        args.llm_model = llm_client.model
        return concepts

    if args.concepts:
        return load_concepts(args.concepts)

    raise ValueError(
        "No data source provided. Use --concepts ..., or --corpus ... --auto-concepts, or --sample."
    )


def has_local_source(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "sample", False)
        or getattr(args, "concepts", None)
        or getattr(args, "auto_concepts", False)
    )


def build_reports(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    concepts = resolve_concepts(args)
    concepts.to_csv(out_dir / "resolved_concepts.csv", index=False)
    corpus_text = read_corpus(args.corpus)
    all_texts, text_index = build_texts_for_embedding(
        concepts,
        mode=args.mode,
        corpus_text=corpus_text,
        max_contexts=args.max_contexts,
        context_window=args.context_window,
    )

    _, all_embeddings = build_backend(all_texts, encoder=args.encoder, model_name=args.model)
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
    if getattr(args, "llm_query", None):
        run_info.update(
            {
                "concept_source": "remote_llm",
                "llm_query": args.llm_query,
                "llm_base_url": args.llm_base_url,
                "llm_model": args.llm_model or "auto",
            }
        )

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


def package_status(module_name: str) -> str:
    return "yes" if importlib.util.find_spec(module_name) is not None else "no"


def run_doctor() -> int:
    print("Concept Driver environment")
    print("--------------------------")
    print("Core engine: local embeddings + local report generation")
    print("Needs generative LLM: no")
    print("Offline baseline available: yes (tfidf)")
    print(f"sentence-transformers installed: {package_status('sentence_transformers')}")
    print(f"umap-learn installed: {package_status('umap')}")
    print(f"ripser installed: {package_status('ripser')}")
    print("")
    print("Recommended commands")
    print("  concept-driver sample")
    print("  concept-driver tui")
    print("  concept-driver tui --llm-base-url https://your-service.up.railway.app/v1")
    print("  concept-driver report --llm-base-url https://your-service.up.railway.app/v1 --llm-query hero --out output/hero-report")
    print("  concept-driver report --concepts path/to/concepts.csv --out output/report")
    print("")
    print("Qwen note")
    print("  Qwen is optional and not needed for the current MVP.")
    print("  If we add Qwen later, it should be for a local embedding backend or an interpretation layer, not as a hard dependency.")
    return 0


def maybe_open_report(out_dir: Path, should_open: bool) -> None:
    if not should_open:
        return
    webbrowser.open((out_dir / "index.html").resolve().as_uri())


def print_tui_help() -> None:
    print("Commands")
    print("  /help   show this help")
    print("  /sets   list available concept sets")
    print("  /terms  list available terms")
    print("  /ask    send the rest of the line to the remote LLM")
    print("  /quit   exit the TUI")


def format_query_result(result: QueryResult) -> str:
    lines = [f"Query: {result.query}"]
    if result.known_term:
        sets = ", ".join(result.concept_sets) if result.concept_sets else "-"
        lines.append(f"Known term: yes ({sets})")
    else:
        lines.append("Known term: no")

    lines.append("Nearest neighbors:")
    if not result.has_signal:
        lines.append("  no similarity signal")
        lines.append("  query is outside the current fitted vocabulary or corpus coverage")
    elif not result.neighbors:
        lines.append("  none")
    else:
        for match in result.neighbors:
            language = f" [{match.language}]" if match.language else ""
            lines.append(
                f"  {match.term} ({match.score:.3f}) - {match.concept_set}{language}"
            )

    lines.append("Contexts:")
    for context in result.contexts[:3]:
        lines.append(f"  - {context}")

    return "\n".join(lines)


def build_llm_client(args: argparse.Namespace) -> RemoteLLMClient | None:
    if not getattr(args, "llm_base_url", None):
        return None
    return RemoteLLMClient(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.llm_model,
        system_prompt=args.llm_system,
        timeout_seconds=args.llm_timeout,
    )


def format_llm_result(prompt: str, response: str) -> str:
    lines = [f"LLM Query: {prompt}", "Response:"]
    if response:
        lines.append(response)
    else:
        lines.append("(empty response)")
    return "\n".join(lines)


def run_tui(args: argparse.Namespace) -> int:
    llm_client = build_llm_client(args)
    session = None
    set_names: list[str] = []
    term_count = 0

    if has_local_source(args):
        concepts = resolve_concepts(args)
        session = build_query_session(
            concepts_path=args.concepts,
            corpus_path=args.corpus,
            mode=args.mode,
            encoder=args.encoder,
            model_name=args.model,
            max_contexts=args.max_contexts,
            context_window=args.context_window,
            concepts_df=concepts,
        )
        term_count = len(session.concepts)
        set_names = sorted(session.concepts["concept_set"].astype(str).unique().tolist())
    elif llm_client is None:
        raise ValueError(
            "No TUI backend provided. Use local data (--concepts/--sample/--auto-concepts) or --llm-base-url."
        )

    print("Concept Driver TUI")
    print("------------------")
    if session is not None:
        print(f"Loaded {term_count} terms across {len(set_names)} concept sets.")
        print(f"Mode: {args.mode} | Encoder: {args.encoder}")
    if llm_client is not None:
        print(f"Remote LLM: {llm_client.base_url} | model={llm_client.model or 'auto'}")
    print("Type a word and press enter. Type /help for commands.")

    while True:
        try:
            raw = input("word> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not raw:
            continue
        if raw in {"/quit", "quit", "exit"}:
            return 0
        if raw == "/help":
            print_tui_help()
            continue
        if raw == "/sets":
            if session is None:
                print("No local concept index loaded.")
                continue
            print("Concept sets:")
            for name in set_names:
                print(f"  {name}")
            continue
        if raw == "/terms":
            if session is None:
                print("No local concept index loaded.")
                continue
            print("Terms:")
            for term in session.concepts["term"].astype(str).tolist():
                print(f"  {term}")
            continue
        if raw.startswith("/ask"):
            if llm_client is None:
                print("No remote LLM configured. Pass --llm-base-url.")
                continue
            prompt = raw[4:].strip()
            if not prompt:
                print("Usage: /ask <prompt>")
                continue
            try:
                response = llm_client.ask(prompt)
            except Exception as exc:  # pragma: no cover
                print(f"LLM error: {exc}")
                continue
            print(format_llm_result(prompt, response))
            print("")
            continue

        if session is None and llm_client is not None:
            try:
                response = llm_client.ask(raw)
            except Exception as exc:  # pragma: no cover
                print(f"LLM error: {exc}")
                continue
            print(format_llm_result(raw, response))
            print("")
            continue

        assert session is not None
        result = session.query(raw, top_k=args.top_k)
        print(format_query_result(result))
        if llm_client is not None and not result.has_signal:
            print("")
            print("Local index has no signal. Try: /ask " + raw)
        print("")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "doctor":
            return run_doctor()
        if args.command == "tui":
            return run_tui(args)

        out_dir = build_reports(args)
        maybe_open_report(out_dir, args.open)
        print(f"Wrote reports to {out_dir}")
        print(f"Open: {out_dir / 'index.html'}")
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
