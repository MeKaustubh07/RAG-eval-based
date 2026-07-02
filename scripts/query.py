"""Query the indexes from the terminal.

    python scripts/query.py "your question"
    python scripts/query.py "your question" --strategy bm25
    python scripts/query.py "your question" --strategy rerank --k 3

Strategies: dense | bm25 | hybrid | rerank
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import STRATEGIES, Pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=STRATEGIES, default="hybrid")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    pipeline = Pipeline(PROJECT_ROOT / "data" / "processed")
    results = pipeline.retrieve(args.question, strategy=args.strategy, k=args.k)

    print(f"\nquery: {args.question}   [strategy={args.strategy}]\n")
    for rank, (chunk_id, score) in enumerate(results, start=1):
        chunk = pipeline.get_chunk(chunk_id)
        print(f"#{rank}  score={score:.3f}  [{chunk_id}]")
        print(f"    {chunk.text[:250]}...\n")


if __name__ == "__main__":
    main()
