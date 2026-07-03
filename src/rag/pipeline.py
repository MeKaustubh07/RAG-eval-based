"""Retrieval pipeline — one entry point, four strategies.

The eval harness, the CLI, and (Day 3) the API all call retrieve() so
strategy comparisons are apples-to-apples: same indexes, same chunks,
only the strategy differs.

Strategies:
  dense   — embedding similarity only (Day 1)
  bm25    — keyword match only (Phase 5)
  hybrid  — both, fused with RRF (Phase 6)
  rerank  — hybrid top-20 → cross-encoder → MMR diversity (Phase 7)

Note the funnel widths in `rerank`: fetch 20 candidates per system,
fuse, cross-encode 20, MMR down to k. Wide early (recall), narrow late
(precision).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cache import SemanticCache
from .chunking import Chunk
from .expansion import expand_query
from .generate import LLMProvider, generate_answer, get_provider
from .hybrid import reciprocal_rank_fusion
from .rerank import cross_encoder_rerank, mmr_select
from .sparse_index import SparseIndex
from .vector_index import load_vector_index

STRATEGIES = ["dense", "bm25", "hybrid", "rerank"]


class Pipeline:
    def __init__(self, processed_dir: str | Path):
        # Backend chosen by VECTOR_BACKEND env (faiss default, qdrant optional)
        # — pipeline code below never mentions the concrete store.
        self.dense = load_vector_index(processed_dir)
        self.sparse = SparseIndex.load(processed_dir)
        # chunk_id → Chunk lookup, shared by all strategies.
        self.chunks_by_id = {chunk.id: chunk for chunk in self.dense.chunks}
        self.cache = SemanticCache()

    def warmup(self) -> None:
        """Force the embedding model to load now, not on the first user query.

        The lazy singletons in embeddings.py/rerank.py mean the FIRST request
        after startup pays the model-load cost (seconds). Calling this once at
        API startup moves that cost off the critical path. Cheap insurance."""
        self.dense.search("warmup", k=1)

    def _passes_filter(self, chunk_id: str, filters: dict | None) -> bool:
        """True if the chunk's metadata satisfies every filter (AND logic).

        Supported keys: `source` (exact filename match), plus any key present
        in chunk.metadata. Missing metadata key → chunk excluded. Kept simple:
        equality only; ranges/operators would go here if the corpus grew."""
        if not filters:
            return True
        chunk = self.chunks_by_id[chunk_id]
        for key, wanted in filters.items():
            have = chunk.source if key == "source" else chunk.metadata.get(key)
            if have != wanted:
                return False
        return True

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        k: int = 10,
        expand_with: LLMProvider | None = None,
        filters: dict | None = None,
    ) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] best-first. Scores are only
        comparable WITHIN a strategy, never across strategies.

        If expand_with is given, generate query rephrasings, retrieve for
        each, and RRF-fuse the lists (multi-query retrieval, Phase 7b).
        If filters is given, keep only chunks whose metadata matches
        (metadata filtering, Phase 16)."""
        if expand_with is not None:
            queries = expand_query(query, expand_with)
            lists = [self.retrieve(q, strategy=strategy, k=k, filters=filters) for q in queries]
            return reciprocal_rank_fusion(lists, top_n=k)

        # Over-fetch when filtering so enough survive the filter to fill k.
        fetch = k if not filters else k * 5

        if strategy == "dense":
            results = self.dense.search(query, fetch)
        elif strategy == "bm25":
            results = self.sparse.search(query, fetch)
        elif strategy == "hybrid":
            results = reciprocal_rank_fusion(
                [self.dense.search(query, fetch * 2), self.sparse.search(query, fetch * 2)],
                top_n=fetch,
            )
        else:
            results = None  # rerank handled below (needs its own funnel)

        if results is not None:
            filtered = [(cid, s) for cid, s in results if self._passes_filter(cid, filters)]
            return filtered[:k]

        if strategy == "rerank":
            fused = reciprocal_rank_fusion(
                [self.dense.search(query, 20), self.sparse.search(query, 20)],
                top_n=20,
            )
            # Filter candidates before the expensive cross-encoder — don't
            # spend compute reranking chunks the filter will drop.
            candidates = [
                self.chunks_by_id[chunk_id]
                for chunk_id, _ in fused
                if self._passes_filter(chunk_id, filters)
            ]
            if not candidates:
                return []
            reranked = cross_encoder_rerank(query, candidates, top_n=max(k, 10))
            survivors = [self.chunks_by_id[chunk_id] for chunk_id, _ in reranked]
            return mmr_select(query, survivors, k=k)

        raise ValueError(f"unknown strategy: {strategy!r} (one of {STRATEGIES})")

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunks_by_id[chunk_id]

    def answer(
        self,
        question: str,
        strategy: str = "rerank",
        k: int = 5,
        provider_name: str = "ollama",
        expand: bool = False,
        filters: dict | None = None,
        use_cache: bool = True,
    ) -> "AnswerResult":
        """Full RAG: retrieve → generate grounded answer → return with trace.

        The trace captures per-stage latency and token proxies — this is the
        observability data the API logs and the frontend's latency panel shows.

        A semantic-cache hit (a near-paraphrase of an earlier question) skips
        the whole pipeline and returns the stored answer in milliseconds.
        Cache is bypassed when filters/expansion are set — those change results,
        so a plain-question cache entry wouldn't be valid for them.
        """
        cacheable = use_cache and not filters and not expand
        scope = f"{strategy}:{k}:{provider_name}"  # answers differ across these
        if cacheable:
            hit = self.cache.get(question, scope=scope)
            if hit is not None:
                payload, similarity = hit
                cached = AnswerResult(**payload)
                cached.trace = {**cached.trace, "cache_hit": True,
                                "cache_similarity": round(similarity, 3), "total_ms": 0.0}
                return cached

        provider = get_provider(provider_name)
        expand_with = provider if expand else None

        t0 = time.perf_counter()
        retrieved = self.retrieve(
            question, strategy=strategy, k=k, expand_with=expand_with, filters=filters
        )
        t1 = time.perf_counter()

        chunks = [self.get_chunk(chunk_id) for chunk_id, _ in retrieved]
        answer = generate_answer(question, chunks, provider)
        t2 = time.perf_counter()

        # Map cited source numbers ([1]-based) back to real chunk IDs.
        cited_ids = [chunks[n - 1].id for n in answer.citations if n <= len(chunks)]

        result = AnswerResult(
            question=question,
            strategy=strategy,
            provider=provider_name,
            expanded=expand,
            answer=answer.text,
            citations=answer.citations,
            cited_chunk_ids=cited_ids,
            chunks=[
                {"id": c.id, "source": c.source, "text": c.text, "score": s}
                for (cid, s), c in zip(retrieved, chunks)
            ],
            trace={
                "retrieval_ms": round(1000 * (t1 - t0), 1),
                "generation_ms": round(1000 * (t2 - t1), 1),
                "total_ms": round(1000 * (t2 - t0), 1),
                "prompt_chars": answer.prompt_chars,
                "completion_chars": answer.completion_chars,
                "approx_prompt_tokens": answer.prompt_chars // 4,  # ~4 chars/token
                "approx_completion_tokens": answer.completion_chars // 4,
                "cache_hit": False,
            },
        )
        if cacheable:
            self.cache.put(question, result.to_dict(), scope=scope)
        return result


@dataclass
class AnswerResult:
    question: str
    strategy: str
    provider: str
    expanded: bool
    answer: str
    citations: list[int]
    cited_chunk_ids: list[str]
    chunks: list[dict]
    trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
