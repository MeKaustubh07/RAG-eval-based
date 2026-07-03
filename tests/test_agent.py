"""Agentic-retrieval tests — drive the graph with a scripted LLM so the
control flow (route, grade, retry cycle, reject) is deterministic and offline.
"""

from src.rag.agent import Agent


class ScriptedProvider:
    """Returns queued replies in order — lets us force any graph path."""

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, prompt: str) -> str:
        return self.replies.pop(0) if self.replies else "SUFFICIENT"


def test_happy_path_route_grade_generate(pipeline):
    # router=RETRIEVE, grade=SUFFICIENT, then the generate call.
    agent = Agent(pipeline, provider=ScriptedProvider(
        ["RETRIEVE", "SUFFICIENT", "Grounded answer. [Source 1]"]
    ))
    out = agent.run("what is reciprocal rank fusion?")
    assert out["attempts"] == 1
    assert any("router → retrieve" in s for s in out["trace"])
    assert any("generate" in s for s in out["trace"])
    assert out["answer"]


def test_reject_out_of_scope(pipeline):
    agent = Agent(pipeline, provider=ScriptedProvider(["REJECT"]))
    out = agent.run("what is the weather in Paris?")
    assert out["attempts"] == 0
    assert "outside the knowledge base" in out["answer"]


def test_reformulate_cycle_then_generate(pipeline):
    # router=RETRIEVE, grade=INSUFFICIENT → reformulate, retrieve again,
    # grade=SUFFICIENT → generate. Proves the retry loop fires.
    agent = Agent(pipeline, provider=ScriptedProvider(
        ["RETRIEVE", "INSUFFICIENT", "better search query", "SUFFICIENT", "Answer. [Source 1]"]
    ))
    out = agent.run("how are search lists merged?")
    assert out["attempts"] == 2, "should have retrieved twice (one retry)"
    assert any("reformulate" in s for s in out["trace"])
