# Concept Driver

Concept Driver turns the one-file `concept_visualizer` prototype into a standalone research repo.

The project direction comes from the shared conversation at [chatgpt.com/share/693cbf33-2110-800a-a5a3-257b9d0ea279](https://chatgpt.com/share/693cbf33-2110-800a-a5a3-257b9d0ea279): start with shape in embedding space, then use that shape to investigate multilingual idioms, conceptual overlaps, voids, and possible topological structures such as knots or entanglements.

## Current MVP

The codebase currently supports a practical first phase:

- load concept sets from CSV
- embed either raw terms or extracted corpus contexts
- project them with PCA and UMAP (when available)
- inspect cosine similarity, nearest neighbors, and lightweight persistence summaries
- export HTML reports for each concept set

This is intentionally narrower than the long-term thesis. The repo starts with a usable explorer, not a full multilingual topology lab.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
concept-driver \
  --concepts examples/example_concepts.csv \
  --corpus examples/example_corpus.txt \
  --mode context \
  --encoder tfidf \
  --out output/report
```

Open `output/report/index.html` after the run completes.

To enable the richer optional backends:

```bash
python -m pip install -e '.[analysis,dev]'
```

## Project Layout

- `src/concept_driver/`: packaged analysis and reporting code
- `examples/`: starter concept CSV and corpus
- `tests/`: baseline regression coverage for the MVP
- `docs/research-brief.md`: condensed project framing recovered from the shared chat
- `docs/roadmap.md`: staged path from prototype to thesis

## Research Direction

The current framing is:

- treat multilingual embedding spaces as geometric objects, not only lookup tables
- distinguish between voids, intertwined structures, and entangled structures
- use shapes as discovery signals before interpreting the underlying idioms or concepts
- move from a short-term exploratory build toward a rigorous multilingual research program

## Near-Term Next Steps

- choose an initial 3-5 language pilot set before scaling toward 10 languages
- add multilingual embedding backends and aligned concept datasets
- add a dedicated topology layer beyond the lightweight H1 summary currently exposed
- formalize evaluation criteria for "voids", "intertwined", and "entangled"
