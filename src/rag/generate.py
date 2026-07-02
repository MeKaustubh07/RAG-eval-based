"""LLM providers behind one interface (Phase 9 grows this; born in Day 2
because the eval harness needs an LLM to generate synthetic questions).

LESSON
------
The interface IS the design lesson: everything downstream (testset
generation, answering, judging) depends on `LLMProvider`, never on
"Ollama" or "Gemini" directly. Swap providers = change one string.
This is dependency inversion — the single most transferable pattern here.

Python concept — Protocol: structural typing. Any class with a matching
`generate()` method satisfies LLMProvider; no inheritance required.
(Like interfaces in Go/TypeScript, unlike Java's `implements`.)

Providers:
  ollama  — local llama3.2 via the Ollama HTTP API (localhost:11434).
            Free, private, no rate limits. Default.
  gemini  — gemini-2.5-flash via google-genai. Smarter; free tier is
            rate-limited. Use for quality-sensitive steps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import requests
from dotenv import load_dotenv

from .chunking import Chunk

load_dotenv()  # pull GEMINI_API_KEY from .env into os.environ

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2"
GEMINI_MODEL = "gemini-2.5-flash"


class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        """Prompt in, completion text out."""
        ...


class OllamaProvider:
    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                # temperature 0.3: mostly deterministic, slight variety —
                # question generation benefits from a little randomness.
                "options": {"temperature": 0.3},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()


class GeminiProvider:
    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai  # imported lazily; only needed if used

        self.model = model
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model, contents=prompt
        )
        return (response.text or "").strip()


def get_provider(name: str = "ollama") -> LLMProvider:
    """Provider factory — the one place that knows concrete classes."""
    if name == "ollama":
        return OllamaProvider()
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"unknown provider: {name!r} (use 'ollama' or 'gemini')")


# ---------------------------------------------------------------------------
# Phase 9: grounded answer generation with citations
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are a precise research assistant. Answer the question using ONLY the numbered sources below.

Rules:
- Use only information in the sources. Do not add outside knowledge.
- After each claim, cite the source it comes from using the exact label shown, like [Source 1] or [Source 2][Source 3].
- Ignore any bracketed numbers inside the source text (e.g. "[82]") — those are the document's own references, not your citation labels. Only cite using the [Source N] labels.
- If the sources do not contain the answer, reply exactly: "The provided sources do not answer this question." Do not guess.
- Be concise.

Sources:
{context}

Question: {question}

Answer:"""


@dataclass
class Answer:
    text: str
    citations: list[int]          # source numbers the model actually cited
    prompt_chars: int             # cheap proxy for prompt token count
    completion_chars: int


def build_context(chunks: list[Chunk]) -> str:
    """Render chunks as a numbered list the prompt (and citations) refer to.

    Numbering is 1-based and positional: source [1] is chunks[0]. The API
    layer keeps the same ordering so a citation [n] maps back to a real chunk.
    """
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Source {i}] (from {chunk.source})\n{chunk.text}")
    return "\n\n".join(blocks)


def _extract_citations(text: str, max_source: int) -> list[int]:
    """Pull [Source n] markers the model emitted, keep valid unique ones.

    Matching "Source n" specifically (not bare [n]) avoids collision with
    academic references like "[82]" that appear inside chunk text.
    """
    import re

    seen: list[int] = []
    for match in re.findall(r"\[Source\s+(\d+)\]", text, flags=re.IGNORECASE):
        n = int(match)
        if 1 <= n <= max_source and n not in seen:
            seen.append(n)
    return seen


def generate_answer(
    question: str,
    chunks: list[Chunk],
    provider: LLMProvider,
) -> Answer:
    """Construct a grounded prompt, call the LLM, parse citations."""
    context = build_context(chunks)
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    text = provider.generate(prompt)
    return Answer(
        text=text,
        citations=_extract_citations(text, len(chunks)),
        prompt_chars=len(prompt),
        completion_chars=len(text),
    )


if __name__ == "__main__":
    # Verify step: both providers answer the same trivial prompt.
    for name in ["ollama", "gemini"]:
        try:
            print(f"{name}: {get_provider(name).generate('Reply with exactly: OK')}")
        except Exception as error:
            print(f"{name}: FAILED — {error}")
