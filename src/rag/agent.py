"""Day 5: Agentic retrieval with LangGraph — a self-correcting RAG graph.

WHY A GRAPH (and why this is the HONEST place for LangGraph)
-----------------------------------------------------------
The linear pipeline (retrieve → rerank → generate) needs no framework — wrapping
it in LangGraph would be cargo cult. This module is different: it has genuine
branching and a CYCLE, which is exactly what a graph engine is for.

    ┌────────┐   off-topic    ┌────────┐
    │ router │ ─────────────► │ reject │──► END
    └───┬────┘                └────────┘
        │ answerable
        ▼
    ┌──────────┐      ┌───────┐  good   ┌──────────┐
    │ retrieve │ ───► │ grade │ ──────► │ generate │──► END
    └──────────┘      └───┬───┘         └──────────┘
        ▲                 │ weak & attempts left
        │                 ▼
        │           ┌─────────────┐
        └───────────│ reformulate │   (LLM rewrites the query, loop back)
                    └─────────────┘

The agent decides, per query:
  - router:      is this answerable from the knowledge base, or out of scope?
                 (the same slot where a SQL tool or web tool would plug in —
                 here only the docs tool is real, and we say so.)
  - grade:       an LLM judges whether the retrieved chunks actually answer the
                 question — the "reflection" step.
  - reformulate: on a weak grade, the LLM rewrites the query and RETRIES, up to
                 a cap. This loop is why we need a graph, not an if-statement:
                 the number of iterations isn't known ahead of time.

All decisions run on the local Ollama LLM — no API cost, no rate limits.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .generate import LLMProvider, generate_answer, get_provider
from .pipeline import Pipeline

MAX_ATTEMPTS = 2  # how many times to reformulate + retry before giving up


class AgentState(TypedDict):
    question: str          # original user question (never mutated)
    query: str             # current (possibly reformulated) search query
    attempts: int          # how many retrieve rounds we've done
    chunk_ids: list[str]   # ids from the latest retrieval
    route: str             # "retrieve" | "reject"
    grade: str             # "sufficient" | "insufficient"
    answer: str
    trace: list[str]       # human-readable log of the agent's decisions


class Agent:
    """A self-correcting retrieval agent over the existing Pipeline."""

    def __init__(self, pipeline: Pipeline, provider: LLMProvider | None = None):
        self.pipeline = pipeline
        self.llm = provider or get_provider("ollama")
        self.graph = self._build_graph()

    # ---------- nodes ----------

    def _route(self, state: AgentState) -> AgentState:
        """Decide whether the knowledge base can plausibly answer this."""
        verdict = self.llm.generate(
            "You route questions to a knowledge base about strength training, "
            "supplements (creatine, protein), and information-retrieval systems.\n"
            f'Question: "{state["question"]}"\n'
            "Answer with ONE word: RETRIEVE if the knowledge base could plausibly "
            "answer it, or REJECT if it is clearly unrelated (e.g. weather, news, math)."
        ).strip().upper()
        route = "reject" if "REJECT" in verdict else "retrieve"
        state["route"] = route
        state["trace"].append(f"router → {route}")
        return state

    def _retrieve(self, state: AgentState) -> AgentState:
        state["attempts"] += 1
        results = self.pipeline.retrieve(state["query"], strategy="hybrid", k=5)
        state["chunk_ids"] = [cid for cid, _ in results]
        state["trace"].append(
            f"retrieve (attempt {state['attempts']}) query={state['query']!r} "
            f"→ {len(state['chunk_ids'])} chunks"
        )
        return state

    def _grade(self, state: AgentState) -> AgentState:
        """LLM reflection: do these chunks actually answer the question?"""
        context = "\n\n".join(
            self.pipeline.get_chunk(cid).text[:400] for cid in state["chunk_ids"]
        )
        verdict = self.llm.generate(
            "Judge whether the CONTEXT contains enough information to answer the QUESTION.\n"
            f"QUESTION: {state['question']}\n\nCONTEXT:\n{context}\n\n"
            "Reply with ONE word: SUFFICIENT or INSUFFICIENT."
        ).strip().upper()
        # "INSUFFICIENT" contains "SUFFICIENT", so check the negative first.
        grade = "insufficient" if verdict.startswith("INSUFFICIENT") else "sufficient"
        state["grade"] = grade
        state["trace"].append(f"grade → {grade}")
        return state

    def _reformulate(self, state: AgentState) -> AgentState:
        """Rewrite the query to try to surface better chunks."""
        new_query = self.llm.generate(
            "The previous search query did not retrieve enough relevant information. "
            "Rewrite it to be more effective — add synonyms or rephrase.\n"
            f'Original question: "{state["question"]}"\n'
            f'Previous query: "{state["query"]}"\n'
            "Output ONLY the new search query."
        ).strip().strip('"')
        state["query"] = new_query or state["question"]
        state["trace"].append(f"reformulate → {state['query']!r}")
        return state

    def _generate(self, state: AgentState) -> AgentState:
        chunks = [self.pipeline.get_chunk(cid) for cid in state["chunk_ids"]]
        answer = generate_answer(state["question"], chunks, self.llm)
        state["answer"] = answer.text
        state["trace"].append("generate → answer produced")
        return state

    def _reject(self, state: AgentState) -> AgentState:
        state["answer"] = "This question is outside the knowledge base (strength training, supplements, and retrieval systems)."
        state["trace"].append("reject → refused (out of scope)")
        return state

    # ---------- edges ----------

    def _after_router(self, state: AgentState) -> str:
        return state["route"]

    def _after_grade(self, state: AgentState) -> str:
        if state["grade"] == "sufficient":
            return "generate"
        if state["attempts"] < MAX_ATTEMPTS:
            return "reformulate"
        state["trace"].append("grade insufficient but out of attempts → generate anyway")
        return "generate"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("router", self._route)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("reformulate", self._reformulate)
        g.add_node("generate", self._generate)
        g.add_node("reject", self._reject)

        g.set_entry_point("router")
        g.add_conditional_edges("router", self._after_router,
                                {"retrieve": "retrieve", "reject": "reject"})
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade,
                                {"generate": "generate", "reformulate": "reformulate"})
        g.add_edge("reformulate", "retrieve")   # the cycle
        g.add_edge("generate", END)
        g.add_edge("reject", END)
        return g.compile()

    # ---------- entry point ----------

    def run(self, question: str) -> dict:
        final = self.graph.invoke({
            "question": question, "query": question, "attempts": 0,
            "chunk_ids": [], "route": "", "grade": "", "answer": "", "trace": [],
        })
        return {
            "question": question,
            "answer": final["answer"],
            "chunk_ids": final["chunk_ids"],
            "attempts": final["attempts"],
            "trace": final["trace"],
        }


if __name__ == "__main__":
    from pathlib import Path

    agent = Agent(Pipeline(Path(__file__).parents[2] / "data" / "processed"))
    for q in ["does creatine cause cramps?", "what is the weather in Paris?"]:
        print(f"\n=== {q} ===")
        out = agent.run(q)
        for step in out["trace"]:
            print("  ·", step)
        print("  ANSWER:", out["answer"][:150])
