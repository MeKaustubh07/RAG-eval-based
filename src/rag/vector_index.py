"""The VectorIndex interface — the seam that lets the retriever ignore which
vector store is underneath.

LESSON
------
Dependency inversion, concretely: `pipeline.py` calls `.search(query, k)` and
`.get_chunk(id)`. It never imports FAISS or Qdrant. So swapping the backend is
a config change (VECTOR_BACKEND=faiss|qdrant), not a code change. This is the
single most transferable idea in the whole project — the same pattern that lets
teams move from SQLite to Postgres, or one cloud to another, without touching
business logic.

Both DenseIndex (FAISS, in-process, zero-setup) and QdrantIndex (real vector DB,
runs as a service, scales past memory) satisfy this Protocol structurally — no
inheritance needed (Python structural typing).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable

from .chunking import Chunk


@runtime_checkable
class VectorIndex(Protocol):
    chunks: list[Chunk]

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return [(chunk_id, cosine_score)] for the k nearest chunks."""
        ...

    def get_chunk(self, chunk_id: str) -> Chunk:
        ...


def load_vector_index(directory: str | Path) -> VectorIndex:
    """Factory: build the configured backend. The ONE place that knows the
    concrete classes — everything else depends on the Protocol."""
    backend = os.environ.get("VECTOR_BACKEND", "faiss").lower()
    if backend == "faiss":
        from .dense_index import DenseIndex

        return DenseIndex.load(directory)
    if backend == "qdrant":
        from .qdrant_index import QdrantIndex

        return QdrantIndex.load(directory)
    raise ValueError(f"unknown VECTOR_BACKEND: {backend!r} (use 'faiss' or 'qdrant')")
