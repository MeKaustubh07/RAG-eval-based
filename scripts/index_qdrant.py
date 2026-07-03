"""Load the chunked corpus into a running Qdrant instance.

    docker run -p 6333:6333 qdrant/qdrant          # or: docker compose up qdrant
    python scripts/index_qdrant.py
    VECTOR_BACKEND=qdrant uvicorn app:app           # serve using Qdrant

Reads the same chunks.json the FAISS build produced, so run build_index.py first.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import Chunk
from src.rag.qdrant_index import QdrantIndex


def main() -> None:
    processed = PROJECT_ROOT / "data" / "processed"
    with open(processed / "chunks.json") as f:
        chunks = [Chunk(**d) for d in json.load(f)]
    print(f"indexing {len(chunks)} chunks into Qdrant ...")
    index = QdrantIndex.build(chunks)
    index.save(processed)  # keeps chunks.json in sync
    # sanity query
    hits = index.search("does creatine help performance?", k=3)
    print("sample hits:", hits)


if __name__ == "__main__":
    main()
