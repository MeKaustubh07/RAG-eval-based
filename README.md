# RAG-eval-based — Hybrid Retrieval RAG, built from fundamentals

A production-inspired Retrieval-Augmented Generation platform built **without
LangChain or LlamaIndex** — every retrieval component is hand-written so the
system is fully legible: chunking, BM25, dense vectors, rank fusion, cross-encoder
reranking, MMR, query expansion, grounded generation with citations.

The centerpiece is an **evaluation harness** that measures which retrieval
strategy actually wins — and a **retrieval regression gate** in CI that goes red
if a change quietly degrades retrieval quality.

> Built as a from-scratch learning project: ~1,800 lines of Python, no retrieval
> frameworks, every design decision measured rather than assumed.

---

## What it does

Ask a question over a document corpus; get a grounded, cited answer. Compare two
retrieval strategies side-by-side on the same query, with per-stage latency.

```
ingest (pdf/md/txt) → chunk → ┬─ dense embeddings (FAISS) ─┐
                              └─ BM25 sparse index ─────────┴─ RRF fusion
   → cross-encoder rerank → MMR diversity → grounded LLM answer + [Source N] citations
```

Four retrieval strategies, switchable per request: `dense`, `bm25`, `hybrid`, `rerank`.

---

## Headline result — the evaluation-bias finding

The eval harness generates synthetic questions with an LLM, then scores every
strategy with **recall@k, precision@k, MRR, nDCG, hit-rate**.

The first test set let the question-generator reuse each source chunk's rare
vocabulary — leaking exact tokens into the question and massively flattering BM25
(pure keyword match). Fixing the prompt to force paraphrasing (what a real user
actually types) collapsed BM25's inflated score and revealed the true ranking:

| strategy | recall@5 (biased Qs) | recall@5 (realistic Qs) |
|---|---|---|
| bm25 | **0.825** | **0.450**  ⟵ collapses |
| dense | 0.625 | 0.600 |
| hybrid | 0.825 | 0.475 |
| rerank | 0.600 | 0.575 |

**BM25's recall@5 drops 0.825 → 0.450** once vocabulary stops leaking — a ~0.38
swing that quantifies the vocabulary-mismatch problem and explains *why* dense
retrieval exists. Full analysis + all metrics: [eval/results.md](eval/results.md).

> The takeaway isn't "which strategy is best" — it's that **your evaluation can
> lie to you**, and catching that is the actual retrieval-engineering skill.

---

## Architecture

```
                 ┌──────────── Ingestion ────────────┐
  data/raw/*  →  │ extract text · clean · metadata    │ → chunks (stable IDs, source, page)
                 └───────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │  Dense: MiniLM → FAISS (cosine)      │   ← swappable via interfaces
              │  Sparse: BM25Okapi                   │
              └──────────────────┬──────────────────┘
   query → [expansion] →         │
              ┌──────────────────┴──────────────────┐
              │  RRF fusion → cross-encoder → MMR    │ → top-k chunks
              └──────────────────┬──────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │  prompt + chunks → LLMProvider →     │ → answer + [Source N] citations
              │  (Ollama local · Gemini)  + trace    │   + latency + token counts
              └─────────────────────────────────────┘

  Eval harness  → recall@k / precision / MRR / nDCG per strategy → results.md + .csv
  Observability → per-query latency · tokens · citations → data/logs.jsonl
```

### Key engineering decisions

| Decision | Why |
|---|---|
| **RRF fuses ranks, not scores** | BM25 (unbounded) and cosine (0–1) scores are incomparable; fusing ranks sidesteps normalization entirely. |
| **Two-stage retrieval** | Cheap bi-encoder + BM25 for recall (top-20), expensive cross-encoder for precision (top-5). Same funnel web search uses. |
| **Provider interfaces** (`LLMProvider`, `VectorIndex`, `EmbeddingModel`) | Swap Ollama↔Gemini, FAISS↔Qdrant, MiniLM↔BGE without touching retrieval code — dependency inversion. |
| **Regression gate in CI** | Retrieval quality breaks silently; a golden-question ratchet turns "did it get worse?" into a red check. |
| **`[Source N]` citation labels** | Research PDFs contain their own `[82]`-style refs; distinct labels stop the model conflating them. |

### Deliberately **not** included (and why)

Postgres/Redis/Celery (at 10k chunks, files + in-process work are correct — a DB
here is résumé-driven complexity), OAuth/RBAC (orthogonal to retrieval),
Elasticsearch (same BM25 algorithm without a cluster to operate). Scope scales
*with* need, not ahead of it.

---

## Stack

Python 3.13 · FastAPI · sentence-transformers (MiniLM + ms-marco cross-encoder) ·
FAISS · rank-bm25 · Ollama / Gemini · pytest + ruff · Docker · GitHub Actions.

---

## Quick start

```bash
# 1. environment
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. local LLM (no API key, no rate limits)
brew install ollama && brew services start ollama && ollama pull llama3.2
# (optional) for Gemini instead: cp .env.example .env  and add GEMINI_API_KEY

# 3. add documents, then build the indexes
cp your-docs/*.pdf data/raw/
python scripts/build_index.py

# 4. query from the terminal …
python scripts/query.py "does creatine cause cramping?" --strategy rerank

# 5. … or run the API + A/B comparison UI
uvicorn app:app --reload      # → http://localhost:8000
```

### API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness + index size |
| `POST /ask` | `{question, strategy, k, provider, expand, filters}` → answer + citations + chunks + trace |
| `POST /compare` | same query through two strategies (powers the A/B UI) |

---

## Evaluation & tests

```bash
python eval/make_testset.py            # generate synthetic Q&A (Ollama)
python eval/run_eval.py                # score all strategies → results.md + .csv

pytest                                 # 21 tests: unit + integration + API + regression gate
ruff check src tests app.py            # lint
```

The test suite builds its own index from the committed Markdown corpus into a
temp dir — **no PDFs, no live LLM, no prebuilt index required** — so it runs
anywhere, including CI.

---

## Deploy

```bash
docker compose up --build              # API reaches host Ollama via host.docker.internal
```

CI (`.github/workflows/ci.yml`): **lint → pytest (incl. regression gate) → docker build** on every push.

---

## Project layout

```
src/rag/     ingest · chunking · embeddings · dense_index · sparse_index ·
             hybrid (RRF) · rerank (cross-encoder + MMR) · expansion · generate · pipeline
eval/        make_testset · run_eval · results.md
tests/       unit + integration + API + regression gate
app.py       FastAPI backend        frontend/index.html   A/B comparison UI
Dockerfile · docker-compose.yml · .github/workflows/ci.yml
PLAN.md      full phase-by-phase build roadmap
```

---

## What this demonstrates

Hybrid dense + sparse retrieval · reciprocal rank fusion · cross-encoder reranking ·
MMR diversity · query expansion · grounded citation-based generation · IR evaluation
(recall@k / MRR / nDCG) · **evaluation-bias analysis** · retrieval regression testing ·
provider abstraction · FastAPI · observability · Docker · CI — production AI-system design.
