"""QdrantIndex — the same VectorIndex interface, backed by a real vector DB.

WHY QDRANT (vs the FAISS DenseIndex)
------------------------------------
FAISS is an in-process library: fast, zero-setup, but the index lives in your
process's memory and dies with it. Qdrant is a standalone service (runs in
Docker) that:
  - persists vectors on disk and survives restarts,
  - serves many clients over the network,
  - does payload filtering server-side (metadata filters run in the DB),
  - scales past a single machine's RAM.

Same `search()` / `get_chunk()` surface as DenseIndex, so pipeline.py can't
tell the difference — that's the whole point of the VectorIndex Protocol.

Cosine: we store normalized vectors and set Distance.COSINE, matching the
FAISS inner-product-on-normalized-vectors setup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .embeddings import embed_query, embed_texts

COLLECTION = "rag_chunks"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


class QdrantIndex:
    def __init__(self, client: QdrantClient, chunks: list[Chunk]):
        self.client = client
        self.chunks = chunks
        self._by_id = {c.id: c for c in chunks}

    # ---------- building ----------

    @classmethod
    def build(cls, chunks: list[Chunk], url: str | None = None) -> "QdrantIndex":
        """Embed chunks and upsert them into a fresh Qdrant collection.

        Point IDs are the chunk's row number; the chunk_id + metadata ride
        along as the point payload so filtering can happen server-side later.
        """
        client = QdrantClient(url=url or QDRANT_URL)
        vectors = embed_texts([c.text for c in chunks]).astype(np.float32)
        dim = vectors.shape[1]

        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vectors[i].tolist(),
                    payload={"chunk_id": c.id, "source": c.source},
                )
                for i, c in enumerate(chunks)
            ],
        )
        print(f"[qdrant] upserted {len(chunks)} vectors into '{COLLECTION}'")
        return cls(client, chunks)

    # ---------- searching ----------

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        query_vector = embed_query(query).astype(np.float32).tolist()
        hits = self.client.query_points(
            collection_name=COLLECTION, query=query_vector, limit=k
        ).points
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self._by_id[chunk_id]

    # ---------- persistence ----------

    def save(self, directory: str | Path) -> None:
        """Qdrant persists vectors itself; we only need chunks.json for the
        text/metadata lookup (the same file FAISS writes)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict

        with open(directory / "chunks.json", "w") as f:
            json.dump([asdict(c) for c in self.chunks], f, indent=1)

    @classmethod
    def load(cls, directory: str | Path, url: str | None = None) -> "QdrantIndex":
        with open(Path(directory) / "chunks.json") as f:
            chunks = [Chunk(**d) for d in json.load(f)]
        return cls(QdrantClient(url=url or QDRANT_URL), chunks)
