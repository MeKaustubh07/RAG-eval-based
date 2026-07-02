"""Query the index from the terminal.

    python scripts/query.py "what is an embedding?"

Day 1 version: dense retrieval only. Day 2 adds --strategy bm25|hybrid|rerank.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.dense_index import DenseIndex


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python scripts/query.py "your question"')
        return
    question = sys.argv[1]

    index = DenseIndex.load(PROJECT_ROOT / "data" / "processed")
    results = index.search(question, k=5)

    print(f"\nquery: {question}\n")
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = index.get_chunk(chunk_id)
        print(f"#{rank}  score={score:.3f}  [{chunk_id}]")
        # First 250 chars as a preview — full text lives in the chunk.
        print(f"    {chunk.text[:250]}...\n")


if __name__ == "__main__":
    main()
