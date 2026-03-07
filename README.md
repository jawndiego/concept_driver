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
concept-driver sample
```

Open `output/sample-report/index.html` after the run completes.

For your own data:

```bash
concept-driver report \
  --concepts examples/example_concepts.csv \
  --corpus examples/example_corpus.txt \
  --mode context \
  --encoder tfidf \
  --out output/report
```

Check installed backends:

```bash
concept-driver doctor
```

Open the terminal UI:

```bash
concept-driver tui --sample
```

Then type a word at the prompt, for example `January` or `red`.

For real data, point the TUI at your corpus and auto-build a vocabulary-backed concept list:

```bash
concept-driver tui \
  --corpus data/your_corpus.txt \
  --auto-concepts \
  --min-freq 3 \
  --max-terms 500
```

For real report generation without manually writing a concept CSV:

```bash
concept-driver report \
  --corpus data/your_corpus.txt \
  --auto-concepts \
  --mode context \
  --encoder tfidf \
  --out output/real-report
```

If you already have a curated concept CSV, use `--concepts` instead.

To query a remote OpenAI-compatible model endpoint from the TUI:

```bash
concept-driver tui \
  --llm-base-url https://conceptdriver-production.up.railway.app/v1 \
  --llm-model HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive
```

Inside the TUI:

- type text directly if you are running in LLM-only mode
- use `/ask <prompt>` if you loaded both a local concept index and a remote LLM

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

## How It Is Powered

Right now the project does not need a generative LLM.

- `tfidf`: fully local baseline, works offline, good for testing the pipeline
- `sentence-transformer`: local embedding model, better quality, still not a chat LLM
- optional future backend: a local multilingual model such as Qwen, if we decide we need a stronger embedding or interpretation layer

For the current MVP, adding Qwen is optional. The bottleneck right now is dataset design and multilingual evaluation, not lack of a chat model.

## Railway Self-Hosted Qwen 9B

The root [Dockerfile](/Users/lreyes/Desktop/Github/JawnDiego/concept_driver/Dockerfile) now targets a real self-hosted `llama.cpp` deployment for:

- `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`
- default quantization: `Q4_K_M`

### Files

- [Dockerfile](/Users/lreyes/Desktop/Github/JawnDiego/concept_driver/Dockerfile)
- [railway-qwen-entrypoint.sh](/Users/lreyes/Desktop/Github/JawnDiego/concept_driver/docker/railway-qwen-entrypoint.sh)

### Required Railway Variables

- `MODEL_REPO`: defaults to `HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive`
- `MODEL_FILE`: defaults to `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf`
- `MODEL_DIR`: defaults to `/tmp/models`
- `N_CTX`: defaults to `8192`
- `N_PARALLEL`: defaults to `1`
- `N_THREADS`: optional, defaults to `nproc`
- `HF_TOKEN`: optional, only needed if you want authenticated model download

### Deploy

Railway will automatically use the root [Dockerfile](/Users/lreyes/Desktop/Github/JawnDiego/concept_driver/Dockerfile).

After deploy:

1. Set the service healthcheck path to `/health`
2. Generate a public domain in Railway networking
3. Wait for the model to finish downloading on first boot
4. Point clients at your Railway URL

### Endpoints

- `GET /health`
- `POST /v1/chat/completions`

### Example Request

```bash
curl https://conceptdriver-production.up.railway.app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
    "messages": [
      {"role": "user", "content": "Define hero in one paragraph."}
    ]
  }'
```

### OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-service.up.railway.app/v1",
    api_key="unused-if-you-did-not-add-auth",
)

response = client.chat.completions.create(
    model="HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Railway Notes

- On Railway Hobby, a persistent volume is too small for the `Q4_K_M` file, so the default setup uses ephemeral storage in `/tmp/models`.
- That is fine for a PoC, but the model will re-download on cold boots and redeploys.
- If you want persistence, use a larger volume plan and point `MODEL_DIR` at the mounted volume path.

## Near-Term Next Steps

- choose an initial 3-5 language pilot set before scaling toward 10 languages
- add multilingual embedding backends and aligned concept datasets
- add a dedicated topology layer beyond the lightweight H1 summary currently exposed
- formalize evaluation criteria for "voids", "intertwined", and "entangled"
