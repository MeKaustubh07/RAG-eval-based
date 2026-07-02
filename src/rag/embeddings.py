"""Phase 3: Embeddings — turn text into vectors where meaning = proximity.

LESSON
------
THE core ML idea of this project. An embedding model maps text to a
point in high-dimensional space (384 dims for MiniLM) such that texts
with similar *meaning* land near each other — even with zero shared
words. "How do I reset my password?" and "forgot login credentials"
end up close. That's what makes dense retrieval work where keyword
search fails.

How similarity is measured: cosine similarity = the cosine of the angle
between two vectors. 1.0 = same direction (same meaning), 0 = unrelated,
negative = opposed. We normalize every vector to length 1, which makes
cosine similarity equal to a plain dot product — cheaper to compute and
exactly what FAISS's inner-product index expects.

Model choice: all-MiniLM-L6-v2 — 22M params, 384 dims, runs fast on CPU,
strong quality for its size. Downloaded from HuggingFace on first use
(~90MB, cached in ~/.cache afterwards).

Python concept — module-level cache: loading the model takes seconds and
hundreds of MB of RAM. We load it once and reuse it (lazy singleton).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None  # lazy singleton


def get_model() -> SentenceTransformer:
    """Load the embedding model once, reuse forever."""
    global _model
    if _model is None:
        print(f"[embeddings] loading {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts → array of shape (len(texts), 384).

    normalize_embeddings=True scales each vector to unit length, so
    downstream dot products ARE cosine similarities.
    Batching: the model processes batch_size texts per forward pass —
    much faster than one at a time.
    """
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 100,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string → shape (384,)."""
    return embed_texts([query])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Explicit cosine similarity, for experiments and sanity checks."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    # Verify step: semantic neighbors must beat lexical strangers.
    words = ["cat", "kitten", "car", "The feline slept on the couch."]
    vectors = embed_texts(words)
    print(f"embedding shape: {vectors.shape}  (texts × dimensions)\n")
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            sim = cosine_similarity(vectors[i], vectors[j])
            print(f"  sim({words[i]!r:40s}, {words[j]!r:40s}) = {sim:.3f}")
    # Expect: cat↔kitten high, cat↔feline-sentence decent, cat↔car low.
