# Roadmap

## Phase 1: Stabilize The Prototype

Goal: create a reproducible repo and preserve the current concept explorer as a maintainable baseline.

- package the code under `src/`
- add a CLI and tests
- keep the HTML reporting workflow
- document the research framing so the code has a clear purpose

## Phase 2: Run A Multilingual Pilot

Goal: move from toy English concept sets to the first serious cross-language comparison.

- choose 3-5 languages
- curate a small aligned concept or idiom dataset
- compare raw term embeddings against context-derived embeddings
- inspect where similarity structure and topological summaries agree or disagree

## Phase 3: Formalize Shape Detection

Goal: stop using "knot" only as a metaphor and define analyzable shape classes.

- add persistent diagrams or richer TDA outputs
- define operational criteria for voids, intertwined structures, and entangled structures
- compare multiple projection and graph-building choices
- record failure cases where attractive shapes collapse under parameter changes

## Phase 4: Scale Toward Thesis Work

Goal: expand from exploratory tooling into a defensible research pipeline.

- scale toward an approximately 10-language study
- add stronger multilingual embedding backends and evaluation protocols
- formalize qualitative interpretation of discovered shapes
- connect the computational results to the theoretical frame in language, culture, and topology

## Immediate Build Order

1. keep the current CLI explorer working
2. choose the first multilingual pilot domain
3. add richer topology outputs only after the pilot dataset is in place
4. separate exploratory claims from validated findings
