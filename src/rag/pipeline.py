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

from pathlib import Path

from .chunking import Chunk
from .dense_index import DenseIndex
from .hybrid import reciprocal_rank_fusion
from .rerank import cross_encoder_rerank, mmr_select
from .sparse_index import SparseIndex

STRATEGIES = ["dense", "bm25", "hybrid", "rerank"]


class Pipeline:
    def __init__(self, processed_dir: str | Path):
        self.dense = DenseIndex.load(processed_dir)
        self.sparse = SparseIndex.load(processed_dir)
        # chunk_id → Chunk lookup, shared by all strategies.
        self.chunks_by_id = {chunk.id: chunk for chunk in self.dense.chunks}

    def retrieve(self, query: str, strategy: str = "hybrid", k: int = 10) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] best-first. Scores are only
        comparable WITHIN a strategy, never across strategies."""
        if strategy == "dense":
            return self.dense.search(query, k)

        if strategy == "bm25":
            return self.sparse.search(query, k)

        if strategy == "hybrid":
            return reciprocal_rank_fusion(
                [self.dense.search(query, k * 2), self.sparse.search(query, k * 2)],
                top_n=k,
            )

        if strategy == "rerank":
            fused = reciprocal_rank_fusion(
                [self.dense.search(query, 20), self.sparse.search(query, 20)],
                top_n=20,
            )
            candidates = [self.chunks_by_id[chunk_id] for chunk_id, _ in fused]
            reranked = cross_encoder_rerank(query, candidates, top_n=max(k, 10))
            survivors = [self.chunks_by_id[chunk_id] for chunk_id, _ in reranked]
            return mmr_select(query, survivors, k=k)

        raise ValueError(f"unknown strategy: {strategy!r} (one of {STRATEGIES})")

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunks_by_id[chunk_id]
