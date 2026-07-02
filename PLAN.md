# RAG From Fundamentals — 3-Day Build Plan

A hybrid-search RAG system built from first principles. No LangChain, no LlamaIndex —
every component hand-written so you understand what frameworks hide.

**Pipeline:** ingest docs → chunk → embed → hybrid retrieval (BM25 + dense) → rerank → LLM answer with citations
**Differentiator:** eval harness (recall@k, MRR) proving which retrieval strategy wins, with numbers.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │                INGESTION                │
  data/raw/*.pdf →  │  extract text → clean → chunk (overlap) │ → data/processed/chunks.json
  data/raw/*.md     └─────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                INDEXING                │
                    │  dense: sentence-transformers → FAISS  │
                    │  sparse: tokenize → BM25 index          │
                    └───────────────────┬───────────────────┘
                                        │
   query ──────────►┌───────────────────┴───────────────────┐
                    │               RETRIEVAL                │
                    │  BM25 top-k  ┐                          │
                    │              ├─ RRF fusion → rerank    │ → top chunks
                    │  dense top-k ┘   (cross-encoder + MMR) │
                    └───────────────────┬───────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │               GENERATION               │
                    │  prompt = query + chunks → LLM →       │ → answer + [1][2] citations
                    └────────────────────────────────────────┘

                    ┌────────────────────────────────────────┐
                    │              EVAL HARNESS              │
                    │  synthetic Q&A → run all strategies →  │ → results table
                    │  recall@k, MRR per strategy             │
                    └────────────────────────────────────────┘
```

## Directory Layout

```
RAG/
├── PLAN.md              ← this file
├── requirements.txt
├── .venv/
├── data/
│   ├── raw/             ← put source documents here (PDF, md, txt)
│   └── processed/       ← chunks.json, indexes (generated, gitignored)
├── src/rag/
│   ├── __init__.py
│   ├── ingest.py        ← Phase 1: load + extract text from files
│   ├── chunking.py      ← Phase 2: split text into overlapping chunks
│   ├── embeddings.py    ← Phase 3: text → vectors
│   ├── dense_index.py   ← Phase 4: FAISS vector search
│   ├── sparse_index.py  ← Phase 5: BM25 keyword search
│   ├── hybrid.py        ← Phase 6: RRF fusion of both
│   ├── rerank.py        ← Phase 7: cross-encoder rerank + MMR diversity
│   ├── generate.py      ← Phase 9: LLM answer with citations
│   └── pipeline.py      ← wires everything together
├── eval/
│   ├── make_testset.py  ← Phase 8a: generate synthetic Q&A pairs
│   ├── run_eval.py      ← Phase 8b: recall@k, MRR per strategy
│   └── results.md       ← generated comparison table
├── scripts/
│   ├── build_index.py   ← CLI: ingest + chunk + index everything
│   └── query.py         ← CLI: ask a question from terminal
├── app.py               ← Phase 10: FastAPI backend
├── frontend/
│   └── index.html       ← Phase 11: minimal chat UI
└── notebooks/           ← scratch space for experiments
```

---

## Day 1 — Foundations: text → searchable vectors

Goal by end of day: `python scripts/query.py "some question"` returns relevant chunks.

### Phase 0: Environment (30 min)
- venv with Python 3.13, install requirements.
- **Learn:** why venvs exist (dependency isolation), what pip actually does.

### Phase 1: Ingestion — `src/rag/ingest.py` (1-2 h)
- Read .txt, .md, .pdf from `data/raw/`. PDF via `pypdf`.
- Output: list of `Document(text, source, metadata)`.
- **Learn:** dataclasses, pathlib, why PDF text extraction is messy (layout, headers, hyphenation).
- **Verify:** print first 500 chars of each loaded doc. Garbage in = garbage out; inspect!

### Phase 2: Chunking — `src/rag/chunking.py` (1-2 h)
- Split documents into ~300-token chunks with ~50-token overlap, respecting sentence boundaries.
- Each chunk gets stable ID (`source:chunk_idx`) — eval harness depends on these later.
- **Learn:** what a token is, why chunk (embedding models have input limits + retrieval granularity),
  why overlap (answers straddling chunk boundaries), tradeoff small chunks (precise but no context)
  vs large (context but diluted embedding).
- **Verify:** print 3 consecutive chunks, see the overlap with your own eyes.

### Phase 3: Embeddings — `src/rag/embeddings.py` (1-2 h)
- `sentence-transformers` with `all-MiniLM-L6-v2` (small, fast, good).
- Function: `embed_texts(list[str]) -> np.ndarray` (N × 384).
- **Learn:** embedding = point in 384-dim space where meaning ≈ proximity. Cosine similarity.
  Play in notebook: embed "cat", "kitten", "car" — check pairwise similarities. THE core ML idea of the project.
- **Verify:** similarity("cat","kitten") > similarity("cat","car").

### Phase 4: Dense index — `src/rag/dense_index.py` (1-2 h)
- FAISS `IndexFlatIP` (exact inner-product search) over normalized embeddings = cosine similarity.
- `build(chunks)`, `search(query, k) -> [(chunk_id, score)]`. Persist index + chunk metadata to disk.
- **Learn:** why a vector index (brute force fine at 10k chunks, HNSW/IVF exist for millions),
  normalized inner product == cosine.
- **Milestone:** `scripts/build_index.py` then `scripts/query.py "question"` → top-5 relevant chunks. RAG retrieval works.

## Day 2 — The advanced part: hybrid retrieval + eval harness

Goal by end of day: `eval/results.md` table proving hybrid > either alone.

### Phase 5: BM25 sparse index — `src/rag/sparse_index.py` (1-2 h)
- `rank_bm25` over tokenized chunks. Same interface as dense index.
- **Learn:** TF-IDF intuition → BM25 (term frequency saturation + length normalization).
  Sparse = exact word match (great for names, error codes, jargon); dense = meaning
  (great for paraphrase). They fail differently — that's why hybrid wins.
- **Verify:** query with rare exact term (function name) — BM25 beats dense. Query with paraphrase — dense beats BM25.

### Phase 6: Hybrid fusion — `src/rag/hybrid.py` (1 h)
- Reciprocal Rank Fusion: `score(d) = Σ 1/(60 + rank_i(d))` across both result lists.
- **Learn:** why fuse ranks not raw scores (BM25 scores and cosine live on different scales — incomparable).

### Phase 7: Rerank + MMR — `src/rag/rerank.py` (1-2 h)
- Cross-encoder `ms-marco-MiniLM-L-6-v2`: scores (query, chunk) pairs jointly. Slow but accurate →
  use only on top-20 candidates. Two-stage retrieval: cheap recall, expensive precision.
- MMR diversity: greedily pick chunks relevant to query but dissimilar to already-picked (λ ≈ 0.7).
- **Learn:** bi-encoder vs cross-encoder (separate embeddings vs joint attention), relevance/diversity tradeoff.

### Phase 8: Eval harness — `eval/` (2-3 h) ★ the differentiator
- `make_testset.py`: for ~50 random chunks, LLM generates a question answerable from that chunk
  → `testset.json` of `{question, expected_chunk_id}`.
- `run_eval.py`: for each strategy (bm25 / dense / hybrid / hybrid+rerank) run all questions, compute:
  - **recall@k** — fraction where expected chunk in top k (k = 1, 5, 10)
  - **MRR** — mean of 1/rank of expected chunk
- Output markdown table → `eval/results.md`.
- **Learn:** you can't improve what you don't measure. Synthetic test data generation. IR metrics.
- **Milestone:** table shows hybrid+rerank > hybrid > single-strategy. If not — investigate, that's the learning.

## Day 3 — Generation, API, frontend, polish

Goal by end of day: browser demo + README with results.

### Phase 9: LLM answer with citations — `src/rag/generate.py` (1-2 h)
- Prompt: numbered chunks `[1]..[5]` + question + "cite sources as [n], say 'not in context' if absent".
- Gemini API (`gemini-2.5-flash`), key in `.env` as `GEMINI_API_KEY`.
- **Learn:** grounding, why citations reduce hallucination, context window budgeting.
- **Verify:** ask something NOT in your docs — model must refuse, not invent.

### Phase 10: FastAPI backend — `app.py` (1-2 h)
- `POST /ask {question, strategy}` → `{answer, citations, chunks, timings}`.
- Load models once at startup (lifespan), not per-request.
- **Learn:** what an API is, pydantic validation at the boundary, model loading cost.

### Phase 11: Frontend — A/B comparison UI — `frontend/index.html` (1-2 h)
- Single HTML file, no build step: question box + **side-by-side panels running two
  retrieval strategies on the same query** (e.g. dense vs hybrid+rerank), each showing
  answer, source chunks, and latency. This IS the demo — differences visible live.
- **Learn:** fetch(), how frontend talks to backend, why visual A/B beats metrics tables for demos.

### Phase 11b: Observability (1 h)
- `/ask` response already returns timings; extend to full per-stage breakdown:
  retrieval ms, rerank ms, LLM ms, prompt/completion token counts.
- Log every query to `data/logs.jsonl` — queries, strategy, latency, tokens. Cheap, huge demo value.
- **Learn:** you operate what you measure; p50 vs p95 latency thinking.

### Phase 12: Polish (2 h)
- README: architecture diagram, eval results table, setup instructions, demo GIF.
- The eval table IS the resume line: "hybrid retrieval improved recall@5 by X% over dense-only."

---

## 10/10 upgrade map

Feature suggestions vs this plan:

| Suggested feature | Status |
|---|---|
| Cross-encoder reranking | Already in — Phase 7 |
| Retrieval observability (latency, recall@k, MRR, hit rate, tokens) | Phase 8 (metrics) + Phase 11b (runtime observability) |
| A/B comparison UI, same query across strategies | Phase 11 redesigned around this |
| Multi-format docs (PDF, MD, DOCX, HTML) | PDF/MD/TXT in Phase 1; DOCX (`python-docx`) + HTML (`beautifulsoup4`) = small Phase 1 extension, add Day 3 if time |
| Query expansion (LLM rephrasings) + multi-query fusion | Stretch goal — LLM generates 3 rephrasings, retrieve each, reuse RRF to fuse. ~40 lines on top of Phase 6. Do after Phase 12 if time |
| Contextual compression before LLM | Post-v1 — cut for 3-day scope |
| Streaming responses + source highlighting | Post-v1 — cut for 3-day scope |

Priority if time is short: A/B UI > observability > multi-query expansion. First two are demo-visible; third reuses existing fusion code.

## Rules of engagement (learning mode)

1. Build phases in order — each depends on previous.
2. After each phase run the verify step. Never stack unverified layers.
3. Type/modify code yourself where possible; understand every line before moving on.
4. When stuck > 30 min, ask — but attempt first.
5. Keep `notebooks/` scratchpad open — test ideas interactively there.

## Key concepts checklist (tick as you learn)

- [ ] virtual environments & pip
- [ ] tokens vs words vs characters
- [ ] embeddings & cosine similarity
- [ ] vector search (FAISS)
- [ ] TF-IDF / BM25
- [ ] rank fusion (RRF)
- [ ] bi-encoder vs cross-encoder
- [ ] MMR diversity
- [ ] recall@k, MRR
- [ ] synthetic test-set generation
- [ ] prompt grounding & citations
- [ ] FastAPI request/response cycle
