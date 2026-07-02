"""Phase 2: Chunking — split documents into overlapping pieces.

LESSON
------
Why chunk at all?
1. Embedding models have input limits (~256-512 tokens for MiniLM);
   text beyond that is truncated and simply ignored.
2. Retrieval granularity: if a whole 50-page doc is one vector, a match
   tells you "the answer is somewhere in these 50 pages" — useless.
   Chunks let you retrieve the *paragraph* that answers the question.

The core tradeoff:
- Small chunks  → precise matches, but each chunk lacks surrounding context.
- Large chunks  → more context, but the embedding becomes a blurry average
                  of many topics and matches get worse.
~300 tokens is a solid default; your eval harness (Day 2) lets you TEST
other sizes instead of guessing.

Why overlap? An answer can straddle a boundary. With 50-token overlap,
text near a cut exists in both neighbors, so one of them still contains
the full thought.

Token ≠ word: models see subword tokens ("chunking" → "chunk"+"ing").
Rule of thumb for English: 1 token ≈ 0.75 words. We count words and
convert — close enough for sizing chunks, no tokenizer dependency needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Document

WORDS_PER_TOKEN = 0.75  # rough English average, good enough for sizing


@dataclass
class Chunk:
    """One retrievable unit of text.

    id: stable identifier "source:index" (e.g. "attention.pdf:12").
        The eval harness stores these as ground truth — they must not
        change between runs, hence deterministic chunking.
    """

    id: str
    text: str
    source: str
    index: int
    metadata: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter: break after . ! ? followed by whitespace.

    Imperfect ("Dr. Smith" splits wrongly) but simple and dependency-free.
    Libraries like nltk/spacy do this better; start simple, upgrade only
    if the eval numbers say chunk boundaries are hurting you.
    """
    # Also treat blank lines (paragraph breaks) as boundaries.
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if p and p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_document(
    doc: Document,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Greedily pack whole sentences into chunks of ~target_tokens.

    Algorithm:
    1. Split into sentences (never cut mid-sentence — a half sentence
       embeds poorly).
    2. Add sentences to the current chunk until the budget is exceeded.
    3. Start the next chunk with the last few sentences of the previous
       one (the overlap), then continue.
    """
    target_words = int(target_tokens * WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN)

    sentences = split_sentences(doc.text)
    chunks: list[Chunk] = []
    current: list[str] = []  # sentences in the chunk being built
    current_words = 0

    def flush() -> None:
        """Finalize the current chunk and seed the next with overlap."""
        nonlocal current, current_words
        if not current:
            return
        text = " ".join(current)
        chunks.append(
            Chunk(
                id=f"{doc.source}:{len(chunks)}",
                text=text,
                source=doc.source,
                index=len(chunks),
                metadata={"words": _word_count(text)},
            )
        )
        # Walk backwards collecting sentences until we have enough overlap.
        kept: list[str] = []
        kept_words = 0
        for sentence in reversed(current):
            if kept_words >= overlap_words:
                break
            kept.insert(0, sentence)
            kept_words += _word_count(sentence)
        current = kept
        current_words = kept_words

    for sentence in sentences:
        words = _word_count(sentence)
        if current_words + words > target_words and current:
            flush()
        current.append(sentence)
        current_words += words

    # Don't lose the tail — but if it's only overlap we already emitted, skip.
    if current and (not chunks or current_words > overlap_words):
        text = " ".join(current)
        chunks.append(
            Chunk(
                id=f"{doc.source}:{len(chunks)}",
                text=text,
                source=doc.source,
                index=len(chunks),
                metadata={"words": _word_count(text)},
            )
        )

    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    """Chunk every document; kwargs pass through to chunk_document."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks


if __name__ == "__main__":
    # Verify step: run `python -m src.rag.chunking` and look at two
    # consecutive chunks — you should SEE the repeated overlap text.
    from pathlib import Path

    from .ingest import load_documents

    docs = load_documents(Path(__file__).parents[2] / "data" / "raw")
    chunks = chunk_documents(docs)
    print(f"{len(docs)} document(s) → {len(chunks)} chunks\n")
    for chunk in chunks[:3]:
        print(f"--- {chunk.id} ({chunk.metadata['words']} words) ---")
        print(chunk.text[:400])
        print()
