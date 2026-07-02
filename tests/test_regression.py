"""Retrieval regression gate — the safety net for retrieval quality.

WHY THIS EXISTS
---------------
Retrieval quality is easy to break silently. Change the chunk size, swap the
embedding model, tweak fusion — every metric could quietly drop and nothing
would fail. This test turns "did retrieval get worse?" into a red CI check.

HOW
---
A small set of hand-written questions, each with a substring that MUST appear
in a correctly-retrieved chunk. We assert that hybrid retrieval surfaces the
right content for at least BASELINE fraction of them in the top-k. Hand-written
(not synthetic) so there's zero LLM dependency and zero vocabulary-leak bias.

Substring matching (not exact chunk IDs) keeps the test robust to chunk-boundary
shifts while still catching real retrieval regressions. If you legitimately
improve things, raise BASELINE — it's a ratchet.
"""

# (question, a phrase that a correct chunk must contain — lowercased match)
GOLDEN = [
    ("how are two search result lists combined?", "reciprocal rank fusion"),
    ("what makes keyword search good for exact terms?", "bm25"),
    ("how is meaning represented as numbers?", "embedding"),
    ("what measures similarity between two vectors?", "cosine"),
    ("why split documents before indexing?", "chunk"),
    ("what does a cross-encoder do differently?", "cross-encoder"),
    ("how is retrieval quality measured?", "recall"),
    ("what keeps retrieved results from being redundant?", "mmr"),
]

BASELINE = 0.75  # ratchet: at least this fraction must retrieve the right content
K = 5


def test_retrieval_quality_meets_baseline(pipeline):
    hits = 0
    misses = []
    for question, must_contain in GOLDEN:
        results = pipeline.retrieve(question, strategy="hybrid", k=K)
        texts = " ".join(pipeline.get_chunk(cid).text.lower() for cid, _ in results)
        if must_contain.lower() in texts:
            hits += 1
        else:
            misses.append((question, must_contain))

    score = hits / len(GOLDEN)
    assert score >= BASELINE, (
        f"retrieval regression: hybrid recall {score:.2f} < baseline {BASELINE:.2f}\n"
        f"missed: {misses}"
    )
