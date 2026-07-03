"""Day 5: MCP server — expose the RAG engine as tools for external AI assistants.

WHAT IS MCP
-----------
The Model Context Protocol is an open standard (Anthropic, 2024) that lets AI
assistants call external tools over a uniform interface. By speaking MCP, this
retrieval engine becomes a tool ANY MCP client (Claude Desktop, IDE agents,
other LLM apps) can use — the assistant asks our server to search the corpus,
gets grounded chunks back, and cites them. Our RAG becomes infrastructure other
agents build on.

TOOLS EXPOSED
  - rag_search(query, k, strategy): return the top chunks (id, source, text, score)
  - rag_answer(question, k):        full grounded answer with citations

TRANSPORT
  stdio — the client launches this process and talks over stdin/stdout. Register
  in an MCP client (e.g. Claude Desktop config) as:

    {
      "mcpServers": {
        "rag": {
          "command": "/absolute/path/.venv/bin/python",
          "args": ["/absolute/path/mcp_server.py"]
        }
      }
    }

Run standalone for a smoke test:  python mcp_server.py
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from src.rag.pipeline import Pipeline

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

mcp = FastMCP("rag")

# Loaded lazily on first tool call so `import mcp_server` (and tool listing)
# doesn't pay the model-load cost.
_pipeline: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(PROCESSED_DIR)
    return _pipeline


@mcp.tool()
def rag_search(query: str, k: int = 5, strategy: str = "hybrid") -> list[dict]:
    """Search the knowledge base and return the most relevant chunks.

    strategy: one of dense | bm25 | hybrid | rerank.
    Returns a list of {chunk_id, source, score, text}.
    """
    pipeline = _get_pipeline()
    results = pipeline.retrieve(query, strategy=strategy, k=k)
    return [
        {
            "chunk_id": cid,
            "source": pipeline.get_chunk(cid).source,
            "score": round(score, 4),
            "text": pipeline.get_chunk(cid).text,
        }
        for cid, score in results
    ]


@mcp.tool()
def rag_answer(question: str, k: int = 5) -> dict:
    """Answer a question from the knowledge base, grounded with citations.

    Returns {answer, citations (chunk ids), sources}.
    """
    result = _get_pipeline().answer(question, strategy="rerank", k=k)
    return {
        "answer": result.answer,
        "citations": result.cited_chunk_ids,
        "sources": sorted({c["source"] for c in result.chunks}),
    }


if __name__ == "__main__":
    mcp.run()  # stdio transport
