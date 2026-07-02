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
from typing import Protocol

import requests
from dotenv import load_dotenv

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


if __name__ == "__main__":
    # Verify step: both providers answer the same trivial prompt.
    for name in ["ollama", "gemini"]:
        try:
            print(f"{name}: {get_provider(name).generate('Reply with exactly: OK')}")
        except Exception as error:
            print(f"{name}: FAILED — {error}")
