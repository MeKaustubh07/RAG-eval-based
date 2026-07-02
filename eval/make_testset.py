"""Phase 8a: generate a synthetic test set of (question, expected_chunk_id).

    python eval/make_testset.py            # ollama (default), 40 questions
    python eval/make_testset.py --provider gemini --n 50

LESSON
------
Ground truth by construction: sample a chunk, have an LLM write a question
that THIS chunk answers, record the chunk's ID as the right answer. Now
"did retrieval work?" is checkable mechanically: did the expected chunk
come back, and at what rank?

Known bias worth understanding: LLMs often reuse the chunk's exact words
in the question, which flatters BM25 (exact-match) relative to real users,
who paraphrase. The prompt pushes against this ("do not copy phrases").
Perfect neutrality is impossible — a known-biased measure beats no measure,
as long as you know the bias. Say exactly that in an interview.

Chunks under 80 words are skipped: too little content to ask anything
specific about (headers, reference lists, page furniture).
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.chunking import Chunk
from src.rag.generate import get_provider

QUESTION_PROMPT = """You are building retrieval test data.

Read this passage and write ONE specific question that the passage answers.

Rules:
- The question must be answerable from this passage alone.
- Phrase it the way a curious person would ask, in their own words — do NOT copy distinctive phrases from the passage.
- Do not mention "the passage", "the text", "the study", or "the author".
- Output ONLY the question, nothing else.

Passage:
{chunk_text}"""

MIN_WORDS = 80


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)  # same seed = same testset
    args = parser.parse_args()

    with open(PROJECT_ROOT / "data" / "processed" / "chunks.json") as f:
        chunks = [Chunk(**data) for data in json.load(f)]

    eligible = [chunk for chunk in chunks if chunk.metadata.get("words", 0) >= MIN_WORDS]
    random.seed(args.seed)
    sampled = random.sample(eligible, min(args.n, len(eligible)))
    print(f"{len(chunks)} chunks, {len(eligible)} eligible, sampling {len(sampled)}")

    provider = get_provider(args.provider)
    testset = []
    for i, chunk in enumerate(sampled, start=1):
        question = provider.generate(QUESTION_PROMPT.format(chunk_text=chunk.text))
        # Models sometimes wrap output in quotes or add a label — strip both.
        question = question.strip().strip('"').removeprefix("Question:").strip()
        if not question.endswith("?") or len(question) < 15:
            print(f"  [{i}/{len(sampled)}] skipped (bad output): {question[:60]!r}")
            continue
        testset.append({"question": question, "expected_chunk_id": chunk.id})
        print(f"  [{i}/{len(sampled)}] {chunk.id}: {question}")

    out_path = PROJECT_ROOT / "eval" / "testset.json"
    with open(out_path, "w") as f:
        json.dump(testset, f, indent=2)
    print(f"\nwrote {len(testset)} questions to {out_path}")


if __name__ == "__main__":
    main()
