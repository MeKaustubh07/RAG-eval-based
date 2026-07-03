"""Day 5: Semantic cache — return a cached answer when a NEW question means
the same thing as one already answered.

LESSON
------
An exact-string cache misses "does creatine cause cramps?" vs "can creatine
give me muscle cramps?" — different strings, identical intent. A semantic
cache embeds the query and matches by cosine similarity instead:

    embed(new query) · embed(cached query) > threshold  → cache hit

This reuses the SAME embedding model the retriever already loaded, so a hit
costs one embedding (~ms) and skips the whole retrieve+rerank+LLM pipeline
(seconds). Classic latency/cost win for repeated or rephrased questions.

Threshold is the knob: too low → wrong answers served from cache (false hits);
too high → few hits. 0.92 is conservative — near-paraphrases only. In-memory
and per-process here; a real deployment would back it with Redis and add TTL.
"""

from __future__ import annotations

import numpy as np

from .embeddings import embed_query


class SemanticCache:
    def __init__(self, threshold: float = 0.92):
        self.threshold = threshold
        self._vectors: list[np.ndarray] = []   # unit-normalized query embeddings
        self._scopes: list[str] = []           # strategy/k/provider the entry was answered with
        self._entries: list[dict] = []         # cached payloads

    def get(self, question: str, scope: str = "") -> tuple[dict, float] | None:
        """Return (payload, similarity) on a hit, else None.

        A hit requires BOTH semantic similarity AND a matching scope — a
        cached `dense` answer must not be served for a `bm25` request, since
        the strategy changes the retrieved chunks and the answer."""
        if not self._vectors:
            return None
        query_vec = embed_query(question)
        sims = np.array(self._vectors) @ query_vec  # cosine (all normalized)
        # Mask out entries from a different scope before picking the best.
        masked = [s if self._scopes[i] == scope else -1.0 for i, s in enumerate(sims)]
        best = int(np.argmax(masked))
        if masked[best] >= self.threshold:
            return self._entries[best], float(masked[best])
        return None

    def put(self, question: str, payload: dict, scope: str = "") -> None:
        self._vectors.append(embed_query(question))
        self._scopes.append(scope)
        self._entries.append(payload)

    def __len__(self) -> int:
        return len(self._entries)
