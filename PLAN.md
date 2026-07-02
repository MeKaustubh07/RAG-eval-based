# RAG From Fundamentals → Production — Build Plan

A hybrid-search RAG system built from first principles, then productionized.
No LangChain, no LlamaIndex — every retrieval component hand-written so you
understand what frameworks hide. Production layer (Docker, CI, provider
abstraction, regression tests) added on top once the core is measured and working.

**Pipeline:** ingest docs → chunk → embed → hybrid retrieval (BM25 + dense) →
query expansion → RRF fusion → cross-encoder rerank → MMR → LLM answer with citations
**Differentiator:** eval harness (recall@k, precision@k, MRR, nDCG) proving which
retrieval strategy wins, with numbers — plus regression tests that fail CI if
retrieval quality drops.

**Structure: core first (Days 1-3), production hardening second (Days 4-5).**
Order matters: infra added before the pipeline is measurable = cargo cult.
Infra added after = engineering.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │                INGESTION                │
  data/raw/* ────►  │  extract (pdf/md/txt/docx/html) →       │ → chunks.json
                    │  clean → chunk (overlap, metadata)      │
                    └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                INDEXING                │
                    │  dense: EmbeddingModel → VectorIndex   │   pluggable interfaces:
                    │  sparse: tokenize → BM25 index          │   FAISS now, Qdrant later
                    └───────────────────┬───────────────────┘
                                        │
   query ──► query expansion (N rephrasings, optional)
                    ┌───────────────────┴───────────────────┐
                    │               RETRIEVAL                │
                    │  BM25 top-k  ┐                          │
                    │              ├─ RRF fusion → rerank    │ → top chunks
                    │  dense top-k ┘   (cross-encoder + MMR) │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │               GENERATION               │
                    │  prompt = query + chunks →              │
                    │  LLMProvider (Gemini | Ollama) →        │ → answer + [1][2] citations
                    │  + latency + token counts               │
                    └────────────────────────────────────────┘

                    ┌────────────────────────────────────────┐
                    │              EVAL HARNESS              │
                    │  synthetic Q&A → run all strategies →  │ → results.md + results.csv
                    │  recall@k, precision@k, MRR, nDCG       │ → pytest regression gate
                    └────────────────────────────────────────┘

                    ┌────────────────────────────────────────┐
                    │             OBSERVABILITY              │
                    │  per-stage latency, tokens, cost →     │ → logs.jsonl + latency panel
                    └────────────────────────────────────────┘
```

## Directory Layout

```
RAG/
├── PLAN.md
├── requirements.txt
├── .env                 ← GEMINI_API_KEY (gitignored)
├── data/
│   ├── raw/             ← source documents (pdf, md, txt, docx, html)
│   ├── processed/       ← chunks.json, indexes (generated, gitignored)
│   └── logs.jsonl       ← per-query observability log
├── src/rag/
│   ├── ingest.py        ← Phase 1: load + extract text (+ per-format parsers)
│   ├── chunking.py      ← Phase 2: overlapping chunks w/ document_id, page metadata
│   ├── embeddings.py    ← Phase 3: EmbeddingModel interface (MiniLM default)
│   ├── dense_index.py   ← Phase 4: VectorIndex interface (FAISS impl)
│   ├── sparse_index.py  ← Phase 5: BM25
│   ├── hybrid.py        ← Phase 6: RRF fusion (also reused by multi-query)
│   ├── rerank.py        ← Phase 7: cross-encoder + MMR
│   ├── expansion.py     ← Phase 7b: LLM query expansion / multi-query
│   ├── generate.py      ← Phase 9: LLMProvider interface (Gemini, Ollama impls)
│   └── pipeline.py      ← wires everything, returns answer + trace (timings, tokens)
├── eval/
│   ├── make_testset.py  ← synthetic Q&A generation
│   ├── run_eval.py      ← metrics per strategy → results.md + results.csv
│   └── results.md
├── tests/
│   ├── test_chunking.py     ← unit tests (Day 4)
│   ├── test_retrieval.py
│   └── test_regression.py   ← eval as pytest: fail if recall@5 drops > threshold
├── scripts/
│   ├── build_index.py
│   └── query.py
├── app.py               ← FastAPI: /ask /index /health
├── frontend/index.html  ← A/B comparison UI (React port optional, Day 5+)
├── Dockerfile           ← Day 4
├── docker-compose.yml   ← Day 4
├── .github/workflows/ci.yml ← Day 4: lint → test → docker build
└── notebooks/
```

---

## Day 1 — Foundations: text → searchable vectors  ✅ DONE

### Phase 0: Environment ✅
venv (Python 3.13), requirements installed.
**Learn:** dependency isolation, pip.

### Phase 1: Ingestion — `src/rag/ingest.py` ✅ (extension pending)
.txt/.md/.pdf → `Document`. **Day 3/4 extension:** DOCX (`python-docx`), HTML
(`beautifulsoup4`) — parser registry pattern, one function per format.
**Learn:** dataclasses, pathlib, PDF extraction mess.

### Phase 2: Chunking — `src/rag/chunking.py` ✅ (metadata upgrade pending)
Sentence-aware ~300-token chunks, 50-token overlap, stable IDs.
**Day 2 upgrade (from industry plan):** add `document_id`, `page_number` (PDFs)
to chunk metadata — enables metadata filtering + citations that point at pages.
**Learn:** tokens, chunk-size tradeoff, overlap rationale.

### Phase 3: Embeddings — `src/rag/embeddings.py` ✅ (interface upgrade Day 4)
MiniLM via sentence-transformers, normalized vectors, cosine = dot product.
**Day 4 upgrade:** extract `EmbeddingModel` protocol so BGE/Jina/Voyage swap in
without touching retrieval code.
**Learn:** embeddings = geometry of meaning, cosine similarity, lazy singleton.

### Phase 4: Dense index — `src/rag/dense_index.py` ✅ (interface upgrade Day 4)
FAISS IndexFlatIP, save/load, CLI query working.
**Day 4 upgrade:** extract `VectorIndex` protocol (build/save/load/search) —
FAISS now, Qdrant optional Day 5. Interface IS the lesson; second backend is bonus.
**Learn:** exact vs approximate search, why brute force fine at this scale.

## Day 2 — The advanced part: hybrid retrieval + eval harness

### Phase 5: BM25 — `src/rag/sparse_index.py` (1-2 h)
`rank_bm25`, same search interface as dense.
**Learn:** TF-IDF → BM25 (k1 saturation, b length normalization). Sparse wins on
exact identifiers; dense wins on paraphrase. Different failure modes → hybrid.

### Phase 6: Hybrid RRF — `src/rag/hybrid.py` (1 h)
`score(d) = Σ 1/(60 + rank_i(d))`. Written generic over N result lists —
multi-query reuses it unchanged (Phase 7b).
**Learn:** why fuse ranks not scores (incompatible scales).

### Phase 7: Rerank + MMR — `src/rag/rerank.py` (1-2 h)
Cross-encoder (`ms-marco-MiniLM-L-6-v2`) on top-20 → top-5. MMR diversity λ≈0.7.
**Learn:** bi- vs cross-encoder, two-stage funnel (cheap recall → expensive
precision — same pattern as web search), relevance/diversity tradeoff.

### Phase 8: Eval harness — `eval/` (2-3 h) ★ the differentiator
- `make_testset.py`: Gemini generates question per sampled chunk → `testset.json`
  (~50 pairs; throttled — free tier rate limits).
- `run_eval.py`: strategies bm25 / dense / hybrid / hybrid+rerank ×  metrics:
  - **recall@k** (k = 1, 5, 10), **precision@k**, **MRR**, **nDCG@10**, **hit rate**
- Output: `results.md` (markdown table) + `results.csv` (chartable).
- **Optional (rate-limit budget permitting):** LLM-as-judge answer metrics —
  faithfulness, citation accuracy. Expensive; run on 10-question subset.
- **Learn:** IR metrics, synthetic test data, measure-don't-guess.
- **Milestone:** table proves hybrid+rerank > hybrid > single strategy.

## Day 3 — Generation, API, A/B frontend

### Phase 7b: Query expansion / multi-query — `src/rag/expansion.py` (1 h)
Gemini generates 3 rephrasings → retrieve per query → RRF-fuse all lists
(reuses Phase 6 code). Eval it: does expansion lift recall@5? Add row to results table.
**Learn:** vocabulary mismatch problem, why expansion helps recall.

### Phase 9: LLM answers — `src/rag/generate.py` (1-2 h)
`LLMProvider` interface from day one: `generate(prompt) -> (text, usage)`.
Two impls: **Gemini** (`gemini-2.5-flash`) and **Ollama** (local fallback, free,
no rate limits). Prompt: numbered chunks + citation instruction + refusal instruction.
**Learn:** grounding, citations vs hallucination, context budgeting, provider abstraction.
**Verify:** ask something NOT in docs — must refuse.

### Phase 10: FastAPI — `app.py` (1-2 h)
`POST /ask {question, strategy, filters}` → `{answer, citations, chunks, trace}`.
`trace` = per-stage ms (retrieval/rerank/LLM) + token counts + estimated cost.
`POST /index` re-ingest endpoint, `GET /health`. Models load once via lifespan.
Every request appended to `data/logs.jsonl`.
**Learn:** pydantic validation at boundary, model loading cost, structured logging.

### Phase 11: A/B comparison UI — `frontend/index.html` (1-2 h)
Side-by-side panels: two strategies, same query — answer, sources (expandable),
latency panel per stage. Single HTML file, no build step.
**Learn:** fetch(), why visual A/B beats metric tables in demos.

### Phase 12: README + results (1-2 h)
Architecture diagram, eval table, setup, demo GIF. Resume line comes from
`results.md`: "hybrid+rerank improved recall@5 by X% over dense-only."

## Day 4 — Production hardening (industry-plan adoptions)

### Phase 13: Tests — `tests/` (2 h)
- Unit: chunking (overlap, stable IDs), retrieval interfaces, RRF math.
- API: `POST /ask`, `GET /health` via httpx TestClient.
- **Retrieval regression gate** ★ best idea in the industry plan:
  `test_regression.py` runs eval on testset, asserts recall@5 ≥ stored baseline
  − 0.05. Change chunk size or embedding model → CI tells you if quality dropped.
**Learn:** pytest, fixtures, testing ML systems (thresholds, not exact values).

### Phase 14: Interfaces refactor (1-2 h)
Extract protocols: `EmbeddingModel`, `VectorIndex`, `LLMProvider` (if not done
inline earlier). Config-driven selection (`config.yaml`).
**Learn:** dependency inversion — the actual production skill, worth more than
any individual backend.

### Phase 15: Docker + CI (2-3 h)
- `Dockerfile` (multi-stage, CPU torch), `docker-compose.yml`.
- GitHub Actions: ruff lint → pytest (incl. regression gate) → docker build.
**Learn:** containerization, why CI catches "works on my machine".

### Phase 16: Metadata filtering (1 h)
`filters` param on /ask: `source`, `tag`, `updated_after`. Applied pre-rerank.
**Learn:** retrieval + structured constraints.

## Day 5+ — Optional escalations (pick by interest, each independent)

| Item | Effort | Value |
|---|---|---|
| Qdrant backend behind `VectorIndex` (docker-compose service) | 2-3 h | proves the interface, real vector DB on resume |
| React + TS + Vite frontend port | 4-6 h | full-stack credibility; do only if targeting full-stack roles |
| Streaming responses (SSE) + source highlighting | 2-3 h | demo polish |
| Context compression (LLM summarizes retrieved chunks pre-prompt) | 2 h | eval it — does it hurt faithfulness? |
| Semantic cache (embed query → cosine vs cached queries → return cached answer) | 2 h | neat systems idea, easy |
| Deploy (Fly.io / Render, free tier) | 2-3 h | live URL on resume |
| MCP server exposing retrieval | 3-4 h | timely, differentiating |
| **Agentic retrieval with LangGraph** — LLM decides per-query: search docs vs SQL vs web, retries with reformulated query when results score low. Genuine graph-with-cycles territory, the honest slot for the LangGraph resume keyword (wrapping our linear pipeline in it would be cargo cult) | 4-6 h | agent orchestration + framework keyword, earned |
| GraphRAG / multimodal / long-term memory | days each | separate projects, backlog |

## Deliberately cut (and why — say this in interviews)

- **PostgreSQL + SQLAlchemy + Redis + Celery** — at 10k chunks, JSON + FAISS
  files + in-process work are correct engineering. Adding a DB/queue here is
  résumé-driven complexity. Right answer scales WITH need, not ahead of it.
- **JWT/OAuth/RBAC** — auth is orthogonal to RAG; a static API-key header check
  (10 lines in FastAPI middleware) demonstrates the boundary without a week of
  identity plumbing.
- **Elasticsearch/OpenSearch** — in-process BM25 is the same algorithm without
  a cluster to operate.
- **Milvus + Weaviate + Qdrant all three** — one interface + one real backend
  proves the point.

## Rules of engagement (learning mode)

1. Build phases in order — each depends on previous.
2. After each phase run the verify step. Never stack unverified layers.
3. Type/modify code yourself where possible; understand every line before moving on.
4. When stuck > 30 min, ask — but attempt first.
5. Keep `notebooks/` scratchpad open for experiments.

## Key concepts checklist

- [ ] virtual environments & pip
- [ ] tokens vs words vs characters
- [ ] embeddings & cosine similarity
- [ ] vector search (FAISS)
- [ ] TF-IDF / BM25
- [ ] rank fusion (RRF)
- [ ] query expansion / multi-query retrieval
- [ ] bi-encoder vs cross-encoder
- [ ] MMR diversity
- [ ] recall@k, precision@k, MRR, nDCG
- [ ] synthetic test-set generation
- [ ] retrieval regression testing
- [ ] prompt grounding & citations
- [ ] provider abstraction (protocols / dependency inversion)
- [ ] FastAPI request/response cycle
- [ ] structured logging & observability (latency, tokens, cost)
- [ ] Docker & CI pipelines
