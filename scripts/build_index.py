"""Build all indexes from the documents in data/raw/.

Run from the project root:
    python scripts/build_index.py

Pipeline: load documents → chunk → embed → save index to data/processed/.
Re-run whenever you add or change documents in data/raw/.
"""

import sys
from pathlib import Path

# Make `src` importable when running this file as a script.
# (Packaging with pyproject.toml is the grown-up fix; this keeps Day 1 simple.)
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import chunk_documents
from src.rag.dense_index import DenseIndex
from src.rag.ingest import load_documents


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    out_dir = PROJECT_ROOT / "data" / "processed"

    print(f"1/3 loading documents from {raw_dir} ...")
    docs = load_documents(raw_dir)
    print(f"    {len(docs)} document(s): {[d.source for d in docs]}")
    if not docs:
        print("    No documents found! Put .txt/.md/.pdf files in data/raw/")
        return

    print("2/3 chunking ...")
    chunks = chunk_documents(docs, target_tokens=300, overlap_tokens=50)
    print(f"    {len(chunks)} chunks")

    print("3/3 building dense index ...")
    index = DenseIndex.build(chunks)
    index.save(out_dir)

    print("\nDone. Try:  python scripts/query.py \"your question here\"")


if __name__ == "__main__":
    main()
