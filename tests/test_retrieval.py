"""Integration tests for retrieval — uses the fixture index (committed .md corpus).

Slower than the unit tests (loads the embedding model once) but still no
network and no LLM. Verifies each strategy returns sane results and that
metadata filtering actually constrains the output.
"""

import pytest

from src.rag.pipeline import STRATEGIES


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_strategy_returns_results(pipeline, strategy):
    results = pipeline.retrieve("what is reciprocal rank fusion?", strategy=strategy, k=5)
    assert results, f"{strategy} returned nothing"
    assert len(results) <= 5
    # Each result is (chunk_id, score) and the id must resolve to a real chunk.
    for chunk_id, _score in results:
        assert pipeline.get_chunk(chunk_id)


def test_bm25_finds_exact_rare_term(pipeline):
    # "IndexFlatIP" appears verbatim in the embeddings explainer — BM25's home turf.
    results = pipeline.retrieve("IndexFlatIP", strategy="bm25", k=3)
    assert results, "BM25 should find an exact rare token"


def test_metadata_filter_restricts_source(pipeline):
    target = "bm25_and_sparse_retrieval.md"
    results = pipeline.retrieve(
        "how does keyword search work?",
        strategy="hybrid",
        k=5,
        filters={"source": target},
    )
    assert results, "filtered retrieval returned nothing"
    for chunk_id, _ in results:
        assert pipeline.get_chunk(chunk_id).source == target, "filter leaked other sources"


def test_filter_with_no_matches_returns_empty(pipeline):
    results = pipeline.retrieve(
        "anything", strategy="dense", k=5, filters={"source": "does_not_exist.md"}
    )
    assert results == []
