"""Unit tests for chunking — pure logic, no models, runs in milliseconds.

These lock in the behaviors the rest of the system depends on: stable IDs
(the eval harness stores them), overlap (answers straddling boundaries),
and sentence integrity.
"""

from src.rag.chunking import chunk_document, split_sentences
from src.rag.ingest import Document


def _doc(text: str) -> Document:
    return Document(text=text, source="test.md")


def test_chunk_ids_are_stable_and_unique():
    doc = _doc(" ".join(f"Sentence number {i} has content." for i in range(60)))
    chunks = chunk_document(doc, target_tokens=60, overlap_tokens=10)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique"
    assert all(c.id == f"test.md:{i}" for i, c in enumerate(chunks)), "IDs must be source:index"


def test_chunking_is_deterministic():
    doc = _doc(" ".join(f"This is fact {i} about a topic." for i in range(40)))
    a = [c.text for c in chunk_document(doc, target_tokens=50, overlap_tokens=10)]
    b = [c.text for c in chunk_document(doc, target_tokens=50, overlap_tokens=10)]
    assert a == b, "same input + params must yield identical chunks"


def test_overlap_shares_text_between_neighbors():
    doc = _doc(" ".join(f"Distinct sentence {i} here." for i in range(50)))
    chunks = chunk_document(doc, target_tokens=40, overlap_tokens=15)
    assert len(chunks) >= 2
    # Some sentence at the tail of chunk 0 should reappear at the head of chunk 1.
    first_sentences = set(split_sentences(chunks[0].text))
    second_sentences = set(split_sentences(chunks[1].text))
    assert first_sentences & second_sentences, "consecutive chunks must overlap"


def test_short_document_yields_one_chunk():
    chunks = chunk_document(_doc("A single short sentence."), target_tokens=300)
    assert len(chunks) == 1
    assert chunks[0].text == "A single short sentence."


def test_split_sentences_breaks_on_terminators():
    parts = split_sentences("First one. Second one! Third one?")
    assert len(parts) == 3
