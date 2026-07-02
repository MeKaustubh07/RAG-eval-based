"""Phase 5: BM25 sparse index — keyword search that complements embeddings.

LESSON
------
"Sparse" because each document is conceptually a vocabulary-sized vector
that is almost all zeros (non-zero only at words it contains) — versus the
"dense" 384-dim embeddings where every dimension carries signal.

BM25 in one breath: score a document for a query term by
  term frequency        — how often the term appears in the document,
  × inverse doc freq    — how rare the term is across the whole corpus,
with two corrections TF-IDF lacks:
  k1 (saturation)       — the 50th occurrence adds almost nothing over the 5th;
                          relevance doesn't scale linearly with repetition,
  b  (length norm)      — a term hit in a 100-word doc is stronger evidence
                          than the same hit in a 10,000-word doc.

Why keep BM25 next to embeddings? They fail on DIFFERENT queries:
  - "IndexFlatIP"        → BM25 exact-matches the token; embeddings see
                           "generic technical string".
  - "forgot my login"    → embeddings match "reset password" docs; BM25
                           shares zero words and returns junk.
Hybrid fusion (Phase 6) gets both behaviors.

No persistence file: building BM25 over a few hundred chunks takes
milliseconds, so we rebuild from chunks.json at load time. Persist things
that are expensive to recompute (embeddings), not things that are free.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from .chunking import Chunk


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer.

    BM25 matches tokens EXACTLY, so tokenization choices are retrieval
    choices: lowercasing makes "Creatine" match "creatine"; keeping
    digits+letters together keeps "vo2max" or error codes intact.
    (Stemming — "supplements" → "supplement" — would help further;
    left out to keep the behavior transparent. Eval can tell you if
    it's worth adding.)
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseIndex:
    """BM25 search with the same interface shape as DenseIndex."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        # k1=1.5, b=0.75 are the library defaults — the standard values.
        self.bm25 = BM25Okapi(corpus_tokens)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] for the k best keyword matches.

        Note: BM25 scores are unbounded and corpus-dependent (unlike
        cosine's [-1, 1]) — comparing them to dense scores directly is
        meaningless. That's exactly why Phase 6 fuses RANKS, not scores.
        """
        scores = self.bm25.get_scores(tokenize(query))
        # argsort ascending → take last k, reversed for descending order.
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i].id, float(scores[i])) for i in top if scores[i] > 0]

    @classmethod
    def load(cls, directory: str | Path) -> "SparseIndex":
        """Rebuild from the chunks.json that DenseIndex.save() wrote."""
        with open(Path(directory) / "chunks.json") as f:
            chunks = [Chunk(**data) for data in json.load(f)]
        return cls(chunks)


if __name__ == "__main__":
    # Verify step: exact-term query → BM25 shines; paraphrase → BM25 flops.
    from pathlib import Path

    index = SparseIndex.load(Path(__file__).parents[2] / "data" / "processed")
    for query in ["IndexFlatIP", "muscle phosphocreatine resynthesis", "getting stronger at the gym"]:
        print(f"\nquery: {query!r}")
        for chunk_id, score in index.search(query, k=3):
            print(f"  {score:7.2f}  {chunk_id}")
