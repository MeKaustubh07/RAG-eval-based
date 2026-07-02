"""Phase 1: Ingestion — load raw files and extract plain text.

LESSON
------
Every RAG system starts here, and it's the least glamorous but most
consequential step: if the extracted text is garbage (broken words,
page headers glued mid-sentence, missing sections), every downstream
stage — chunking, embedding, retrieval — inherits that garbage.
Always eyeball the output of this stage before building on it.

Python concepts used:
- @dataclass: auto-generates __init__/__repr__ for classes that just hold data.
- pathlib.Path: object-oriented file paths (better than string manipulation).
- Generators would work too, but we return lists for simplicity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

# File types we know how to read, mapped to the function that reads them.
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Document:
    """One source file, reduced to plain text.

    text:     the full extracted text
    source:   filename it came from (becomes part of chunk IDs later,
              and is what citations point back to)
    metadata: anything extra worth keeping (page count, title, ...)
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _read_text_file(path: Path) -> str:
    """.txt and .md are already plain text — just read them.

    errors="replace" swaps undecodable bytes for � instead of crashing;
    real-world files often have encoding quirks.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF, page by page.

    PDF is a *layout* format, not a text format — it stores "draw these
    glyphs at these coordinates". pypdf reconstructs reading order, which
    mostly works but can produce artifacts: hyphenated line breaks,
    repeated page headers, mangled tables. Inspect the output!
    """
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_documents(raw_dir: str | Path) -> list[Document]:
    """Load every supported file under raw_dir into a Document.

    Sorted so runs are deterministic — same input, same output order,
    same chunk IDs. Determinism matters once the eval harness stores
    expected chunk IDs.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    documents: list[Document] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = _read_text_file(path)

        text = clean_text(text)
        if not text.strip():
            print(f"  [warn] no text extracted from {path.name}, skipping")
            continue

        documents.append(
            Document(
                text=text,
                source=path.name,
                metadata={"path": str(path), "chars": len(text)},
            )
        )

    return documents


def clean_text(text: str) -> str:
    """Light cleanup — collapse excessive blank lines, strip trailing spaces.

    Deliberately conservative: aggressive cleaning (regex-stripping
    "noise") tends to eat real content. Clean only what you've verified
    is noise for YOUR documents.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run > 2:  # allow at most 2 consecutive blank lines
                continue
        else:
            blank_run = 0
        cleaned.append(line)
    return "\n".join(cleaned)


if __name__ == "__main__":
    # Verify step: run `python -m src.rag.ingest` from the project root
    # and READ the output. Does the text look like the document?
    docs = load_documents(Path(__file__).parents[2] / "data" / "raw")
    print(f"Loaded {len(docs)} document(s)\n")
    for doc in docs:
        print(f"=== {doc.source} ({doc.metadata['chars']} chars) ===")
        print(doc.text[:500])
        print("...\n")
