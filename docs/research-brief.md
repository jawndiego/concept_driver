# Research Brief

## Source

This brief condenses the shared conversation at:

- [chatgpt.com/share/693cbf33-2110-800a-a5a3-257b9d0ea279](https://chatgpt.com/share/693cbf33-2110-800a-a5a3-257b9d0ea279)

## Working Thesis

Multilingual embedding spaces may contain meaningful geometric structure that is not reducible to nearest-neighbor semantics alone. Those structures can include:

- voids: meaningful absences or discontinuities
- knots or other dense shapes: possible sites of cultural overlap, idiomatic compression, or conceptual ambiguity
- intertwined structures: concepts that remain separable but stay tightly linked
- entangled structures: concepts whose meaning emerges through relation and is difficult to isolate cleanly

The central methodological inversion is important: detect shape first, then inspect the idiom, concept, or cultural boundary that the shape may be signaling.

## What The Shared Conversation Clarified

- The project should combine theory and implementation rather than choosing one side.
- The first real milestone is exploratory, but it should ladder into thesis-grade work.
- A likely target is an approximately 10-language study, but the language set is still open.
- 3D projection is a visualization aid, not the complete analysis method.
- Topological analysis is useful because shape can persist even when token labels are ignored.

## MVP Boundary For This Repo

This repository currently implements the first practical layer:

- concept-set ingestion from CSV
- context extraction from plain-text corpora
- term or context embedding
- PCA, UMAP fallback, similarity heatmaps, kNN tables
- lightweight persistence summaries when `ripser` is installed

That makes the repo useful now while leaving room for the stronger multilingual and topological work later.

## Open Research Questions

- Which shapes are stable enough to treat as signals rather than visualization artifacts?
- How should "intertwined" and "entangled" be operationalized in data terms?
- What counts as a void in a multilingual embedding manifold?
- Can shape-based search surface previously unmodeled idioms or culture-specific concepts?
- Which multilingual models preserve enough nuance for this work?

## Suggested First Pilot

- pick 3-5 languages before scaling to 10
- choose 1-2 tightly bounded concept domains such as emotions, kinship, time, or praise/insult idioms
- compare term mode against context mode
- treat topology as one analysis layer, not the only one
- keep the raw contexts because interpretation depends on them
