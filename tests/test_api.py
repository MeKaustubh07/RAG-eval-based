"""API tests — exercise the HTTP layer without a live LLM.

We monkeypatch the provider so /ask and /compare run fast and offline:
the test asserts the ENDPOINT wiring (validation, response shape, logging),
not the LLM's answer quality. Retrieval still runs for real against the
committed .md corpus.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


class FakeProvider:
    """Deterministic stand-in for Ollama/Gemini — no network."""

    def generate(self, prompt: str) -> str:
        return "This is a grounded test answer. [Source 1]"


@pytest.fixture
def client(monkeypatch, tmp_path):
    import app as app_module
    from src.rag import pipeline as pipeline_module

    # Point the app at an index built from the committed .md files.
    from src.rag.chunking import chunk_documents
    from src.rag.dense_index import DenseIndex
    from src.rag.ingest import load_documents

    docs = [d for d in load_documents(PROJECT_ROOT / "data" / "raw") if d.source.endswith(".md")]
    processed = tmp_path / "processed"
    DenseIndex.build(chunk_documents(docs, target_tokens=200, overlap_tokens=40)).save(processed)

    monkeypatch.setattr(app_module, "PROCESSED_DIR", processed)
    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "logs.jsonl")
    # Every get_provider() call (in generate + expansion) returns the fake.
    monkeypatch.setattr(pipeline_module, "get_provider", lambda name="ollama": FakeProvider())

    with TestClient(app_module.app) as c:  # triggers lifespan (loads pipeline)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert "rerank" in body["strategies"]


def test_ask_returns_answer_and_trace(client):
    resp = client.post("/ask", json={"question": "what is BM25?", "strategy": "hybrid", "k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]
    assert body["citations"] == [1]           # FakeProvider cited [Source 1]
    assert len(body["chunks"]) <= 3
    assert "total_ms" in body["trace"]


def test_ask_rejects_too_short_question(client):
    # min_length=3 → pydantic 422 before our code runs.
    assert client.post("/ask", json={"question": "hi"}).status_code == 422


def test_compare_runs_two_strategies(client):
    resp = client.post(
        "/compare",
        json={"question": "how does hybrid search work?", "strategy_a": "dense", "strategy_b": "bm25", "k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["a"]["strategy"] == "dense"
    assert body["b"]["strategy"] == "bm25"
