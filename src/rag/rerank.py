"""Phase 7: Cross-encoder reranking + MMR diversity.

LESSON
------
Two-stage retrieval, the pattern behind every serious search system:

  stage 1 (RECALL, cheap):    BM25 + dense + RRF cast a wide net → top ~20.
                              Bi-encoder: query and docs embedded SEPARATELY,
                              similarity is just a dot product. Fast because
                              doc vectors are precomputed — but the model
                              never reads query and doc together.

  stage 2 (PRECISION, costly): cross-encoder re-scores each (query, doc)
                              PAIR in one forward pass. Its attention layers
                              see both texts jointly, catching interactions
                              a dot product can't ("no significant effect
                              was found" vs "significant effect found" embed
                              nearly identically; a cross-encoder reads the
                              negation). Too slow to run on the whole corpus
                              — affordable on 20 candidates.

MMR (Maximal Marginal Relevance) — the second problem: top-k results are
often five near-copies of the same passage. Redundant context wastes the
LLM's context window. MMR picks greedily:

    next = argmax  λ·relevance(query, d)  −  (1−λ)·max_sim(d, already_picked)

λ = 1.0 → pure relevance (plain top-k). λ = 0.0 → pure diversity (drifts
off-topic). 0.7 is a sane default: mostly relevant, penalize duplicates.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from .chunking import Chunk
from .embeddings import embed_query, embed_texts

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder: CrossEncoder | None = None  # lazy singleton, like embeddings


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        print(f"[rerank] loading {CROSS_ENCODER_MODEL} ...")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def cross_encoder_rerank(
    query: str,
    chunks: list[Chunk],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Re-score candidate chunks against the query, jointly.

    Input order doesn't matter — every (query, chunk) pair gets an
    independent relevance score. Higher = more relevant (raw logits,
    not probabilities; only the ordering matters).
    """
    model = get_cross_encoder()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [(chunk.id, float(score)) for chunk, score in ranked[:top_n]]


def mmr_select(
    query: str,
    chunks: list[Chunk],
    k: int = 5,
    lambda_mult: float = 0.7,
) -> list[tuple[str, float]]:
    """Greedy MMR: relevant to the query, dissimilar to already-picked.

    Re-embeds the candidates (≤ ~20 texts — cheap) rather than plumbing
    vectors out of FAISS; clarity over micro-optimization at this scale.
    Returns [(chunk_id, mmr_score)] in selection order.
    """
    if not chunks:
        return []
    query_vec = embed_query(query)
    chunk_vecs = embed_texts([chunk.text for chunk in chunks])

    # All vectors are unit-normalized → dot product = cosine similarity.
    relevance = chunk_vecs @ query_vec               # (n,) sim to query
    pairwise = chunk_vecs @ chunk_vecs.T             # (n, n) sim between chunks

    selected: list[int] = []
    remaining = list(range(len(chunks)))
    while remaining and len(selected) < k:
        best_idx, best_score = None, -np.inf
        for i in remaining:
            redundancy = max((pairwise[i][j] for j in selected), default=0.0)
            score = lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [(chunks[i].id, float(lambda_mult * relevance[i])) for i in selected]
