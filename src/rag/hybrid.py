"""Phase 6: Reciprocal Rank Fusion — merge result lists from different systems.

LESSON
------
Problem: BM25 returns scores like 12.7, dense returns cosine like 0.79.
Different scales, different distributions — adding or averaging them is
statistically meaningless. You could try normalizing scores, but score
distributions shift with every query, so normalization is fragile.

RRF's move: throw the scores away entirely, keep only the RANKS.

    fused_score(doc) = Σ over result lists:  1 / (k + rank(doc))

with k = 60 (the value from the original 2009 paper; surprisingly robust).

Why it works:
- rank 1 contributes 1/61, rank 10 contributes 1/70 — smooth decay, no cliff.
- k=60 dampens the head: being rank 1 vs rank 3 matters, but doesn't dominate.
  A document ranked 3rd by BOTH systems beats one ranked 1st by only one.
- Zero parameters to tune (k barely matters in practice), no training.
  Production systems at large companies still use it. Simple survives.

Written to take ANY number of result lists — Phase 7b (multi-query
retrieval) fuses 4 lists with this exact function, unchanged.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked result lists into one, by reciprocal rank.

    result_lists: each is [(chunk_id, score)] sorted best-first.
                  Incoming scores are IGNORED — only positions matter.
    Returns [(chunk_id, fused_score)] sorted best-first.
    """
    fused: dict[str, float] = {}
    for results in result_lists:
        for rank, (chunk_id, _score) in enumerate(results, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return ranked[:top_n] if top_n else ranked


if __name__ == "__main__":
    # Verify step: doc "c" is mid-ranked by BOTH lists and should beat
    # docs that only ONE list liked.
    bm25_results = [("a", 9.1), ("c", 7.0), ("b", 6.2)]
    dense_results = [("d", 0.9), ("c", 0.8), ("e", 0.7)]
    for chunk_id, score in reciprocal_rank_fusion([bm25_results, dense_results]):
        print(f"  {score:.5f}  {chunk_id}")
    # Expect: c first (appears in both), then a/d (rank-1s), then b/e.
