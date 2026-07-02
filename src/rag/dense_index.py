"""Phase 4: Dense index — store chunk vectors, find nearest neighbors fast.

LESSON
------
A vector index answers one question: "given this query vector, which of
my N stored vectors are closest?"

We use FAISS IndexFlatIP:
- "Flat"  = brute force — compares the query against EVERY vector.
            Exact results, no approximation. At our scale (thousands of
            chunks) this takes microseconds. Approximate indexes
            (HNSW, IVF) only matter at millions of vectors, where they
            trade a little recall for a lot of speed.
- "IP"    = inner product. Because our embeddings are normalized to unit
            length (see embeddings.py), inner product == cosine similarity.

Persistence: building embeddings is the slow part (model inference over
every chunk), so we do it once in build_index.py and save to disk:
  - dense.faiss   — the FAISS index (binary)
  - chunks.json   — chunk texts + IDs, positions aligned with the index
FAISS returns *row numbers*; chunks.json maps row number → chunk.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk
from .embeddings import embed_query, embed_texts


class DenseIndex:
    """Cosine-similarity search over chunk embeddings."""

    def __init__(self, index: faiss.Index, chunks: list[Chunk]):
        self.index = index
        self.chunks = chunks  # row i in the index ↔ chunks[i]

    # ---------- building ----------

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "DenseIndex":
        """Embed every chunk and add the vectors to a fresh index."""
        print(f"[dense] embedding {len(chunks)} chunks ...")
        vectors = embed_texts([chunk.text for chunk in chunks])
        dim = vectors.shape[1]  # 384 for MiniLM
        index = faiss.IndexFlatIP(dim)
        index.add(vectors.astype(np.float32))  # FAISS wants float32
        return cls(index, chunks)

    # ---------- searching ----------

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Return [(chunk_id, score)] for the k most similar chunks.

        FAISS search is batched: it expects a 2D array of queries and
        returns 2D arrays of (scores, row_indices). We send one query
        and take row [0] of each.
        """
        query_vector = embed_query(query).astype(np.float32).reshape(1, -1)
        scores, rows = self.index.search(query_vector, k)
        return [
            (self.chunks[row].id, float(score))
            for row, score in zip(rows[0], scores[0])
            if row != -1  # FAISS pads with -1 when k > number of vectors
        ]

    def get_chunk(self, chunk_id: str) -> Chunk:
        """Look up a chunk by its ID (linear scan is fine at this scale)."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        raise KeyError(chunk_id)

    # ---------- persistence ----------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "dense.faiss"))
        with open(directory / "chunks.json", "w") as f:
            json.dump([asdict(chunk) for chunk in self.chunks], f, indent=1)
        print(f"[dense] saved index ({self.index.ntotal} vectors) to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "DenseIndex":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "dense.faiss"))
        with open(directory / "chunks.json") as f:
            chunks = [Chunk(**data) for data in json.load(f)]
        return cls(index, chunks)
