"""Phase 7b: Query expansion / multi-query retrieval.

LESSON
------
The vocabulary-mismatch problem, made concrete by our own eval: a user
asks "how do muscles recharge energy between sets?" but the document says
"phosphocreatine resynthesis". Different words, same meaning — dense
retrieval bridges some of this, but not all.

Query expansion attacks it head-on: ask an LLM to rewrite the question
several ways, retrieve for EACH rewrite, then fuse all the result lists
with the same RRF from Phase 6. A chunk that any phrasing surfaces gets a
chance; chunks that multiple phrasings agree on rise to the top.

This is why hybrid.py was written generic over N lists — multi-query just
hands it more lists. No new fusion code.

Cost: N extra retrievals + one LLM call per query. Worth it when recall
matters more than latency (most RAG). The eval harness tells you if it
actually helps on YOUR corpus — don't assume, measure.
"""

from __future__ import annotations

from .generate import LLMProvider

EXPANSION_PROMPT = """A user asked this question of a search system:

"{query}"

Write {n} alternative phrasings that a search engine could use to find
relevant documents. Vary the vocabulary — use synonyms and rephrase
technical terms in plain language, and vice versa. Keep each on its own
line, no numbering, no extra text.

Alternatives:"""


def expand_query(query: str, provider: LLMProvider, n: int = 3) -> list[str]:
    """Return the original query plus n LLM-generated rephrasings.

    Always includes the original — expansion should only ADD recall, never
    lose the user's exact wording. Deduped, capped defensively.
    """
    raw = provider.generate(EXPANSION_PROMPT.format(query=query, n=n))
    variants = [line.strip(" -•\t") for line in raw.splitlines() if line.strip()]

    queries = [query]
    for variant in variants:
        if variant and variant.lower() != query.lower() and variant not in queries:
            queries.append(variant)
    return queries[: n + 1]


if __name__ == "__main__":
    from .generate import get_provider

    for q in expand_query("how do muscles recharge energy between sets?", get_provider("ollama")):
        print(f"  - {q}")
