"""Phase 10: FastAPI backend — serve the RAG pipeline over HTTP.

    uvicorn app:app --reload

Endpoints:
    GET  /health          liveness + index size
    POST /ask             run RAG, return answer + chunks + trace
    POST /compare         run TWO strategies on one query (A/B UI)
    GET  /                serve the frontend

LESSON
------
- Models load ONCE at startup (lifespan), not per request. Loading the
  cross-encoder takes seconds and hundreds of MB — doing it per request
  would make every call unusable. The Pipeline lives for the process.
- Pydantic validates the request body at the boundary: bad input is
  rejected with a clear 422 before it reaches our code. Validate at the
  edges, trust the inside.
- Every /ask is appended to data/logs.jsonl — the observability trail.
  You operate what you can see.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.rag.pipeline import STRATEGIES, Pipeline

PROJECT_ROOT = Path(__file__).parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_PATH = PROJECT_ROOT / "data" / "logs.jsonl"

# Populated in the lifespan handler, shared across requests.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] loading pipeline (indexes + models) ...")
    pipeline = Pipeline(PROCESSED_DIR)
    pipeline.warmup()  # load embedding model now, off the first request's path
    state["pipeline"] = pipeline
    from src.rag.agent import Agent
    state["agent"] = Agent(pipeline)
    print(f"[startup] ready — {len(pipeline.chunks_by_id)} chunks")
    yield
    state.clear()


app = FastAPI(title="RAG Platform", version="1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy: str = "rerank"
    k: int = Field(default=5, ge=1, le=20)
    provider: str = "ollama"
    expand: bool = False
    filters: dict | None = None  # e.g. {"source": "CreatineResearchPaper.pdf"}


class CompareRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    strategy_a: str = "dense"
    strategy_b: str = "rerank"
    k: int = Field(default=5, ge=1, le=20)
    provider: str = "ollama"


def _log(record: dict) -> None:
    """Append one JSON line to the observability log."""
    record["ts"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health() -> dict:
    pipeline = state.get("pipeline")
    return {
        "status": "ok" if pipeline else "loading",
        "chunks": len(pipeline.chunks_by_id) if pipeline else 0,
        "strategies": STRATEGIES,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    result = state["pipeline"].answer(
        req.question,
        strategy=req.strategy,
        k=req.k,
        provider_name=req.provider,
        expand=req.expand,
        filters=req.filters,
    )
    payload = result.to_dict()
    _log(
        {
            "question": req.question,
            "strategy": req.strategy,
            "provider": req.provider,
            "expand": req.expand,
            "citations": result.cited_chunk_ids,
            "trace": result.trace,
        }
    )
    return payload


@app.post("/compare")
def compare(req: CompareRequest) -> dict:
    """Run the same query through two strategies — powers the A/B UI."""
    pipeline = state["pipeline"]
    results = {}
    for side, strategy in [("a", req.strategy_a), ("b", req.strategy_b)]:
        result = pipeline.answer(
            req.question, strategy=strategy, k=req.k, provider_name=req.provider
        )
        results[side] = result.to_dict()
    _log(
        {
            "mode": "compare",
            "question": req.question,
            "strategy_a": req.strategy_a,
            "strategy_b": req.strategy_b,
            "trace_a": results["a"]["trace"],
            "trace_b": results["b"]["trace"],
        }
    )
    return results


class AgentRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.post("/agent")
def agent(req: AgentRequest) -> dict:
    """Self-correcting agentic retrieval — routes, grades, retries, generates."""
    result = state["agent"].run(req.question)
    _log({"mode": "agent", "question": req.question,
          "attempts": result["attempts"], "trace": result["trace"]})
    return result


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")
