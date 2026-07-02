"""Shared pytest fixtures.

The retrieval tests need a real index, but the production index (data/processed)
and the PDF corpus are gitignored — CI won't have them. So we build a small,
self-contained index from the committed Markdown explainer files into a temp
directory once per test session. This makes retrieval tests reproducible
anywhere, including CI, with no external data.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import chunk_documents
from src.rag.dense_index import DenseIndex
from src.rag.ingest import load_documents
from src.rag.pipeline import Pipeline


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory) -> Pipeline:
    """Build an index from the committed .md files and return a Pipeline.

    Session-scoped: the embedding model loads once for the whole test run.
    """
    processed = tmp_path_factory.mktemp("processed")

    # Only the committed Markdown files — deterministic, always present.
    raw = PROJECT_ROOT / "data" / "raw"
    docs = [d for d in load_documents(raw) if d.source.endswith(".md")]
    assert docs, "expected committed .md explainer files in data/raw"

    chunks = chunk_documents(docs, target_tokens=200, overlap_tokens=40)
    DenseIndex.build(chunks).save(processed)
    return Pipeline(processed)
