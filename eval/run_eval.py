"""Phase 8b: score every retrieval strategy on the test set.

    python eval/run_eval.py

Outputs eval/results.md (table) and eval/results.csv (chartable).

LESSON — the metrics (one relevant chunk per question, which simplifies all of them)
------
recall@k      Was the expected chunk in the top k? Averaged over questions.
              The workhorse metric: "does the right thing come back at all?"
precision@k   Relevant results in top k / k. With a single relevant chunk the
              ceiling is 1/k — the ABSOLUTE number looks tiny; only compare
              BETWEEN strategies.
MRR           Mean of 1/rank of the expected chunk (0 if absent). Rank 1 → 1.0,
              rank 4 → 0.25. Cares WHERE the hit landed, not just whether.
nDCG@k        DCG = 1/log2(rank+1); ideal DCG = 1 (hit at rank 1), so
              nDCG = DCG here. Gentler position discount than MRR (log vs
              linear). Included because industry dashboards speak nDCG.
hit_rate@k    Same as recall@k in the single-relevant case — kept as its own
              column because job specs name it separately.

Why position matters at all: the LLM reads chunks in order and attends most
reliably to the start of its context. Rank 1 vs rank 9 changes answer quality
even when both "contain" the answer.
"""

import csv
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pipeline import STRATEGIES, Pipeline

K_VALUES = [1, 5, 10]
RETRIEVE_K = 10


def evaluate_strategy(pipeline: Pipeline, testset: list[dict], strategy: str) -> dict:
    """Run every test question through one strategy, aggregate metrics."""
    recall = {k: 0.0 for k in K_VALUES}
    mrr_total = 0.0
    ndcg_total = 0.0
    latencies: list[float] = []

    for item in testset:
        start = time.perf_counter()
        results = pipeline.retrieve(item["question"], strategy=strategy, k=RETRIEVE_K)
        latencies.append(time.perf_counter() - start)

        retrieved_ids = [chunk_id for chunk_id, _ in results]
        # rank of the expected chunk, 1-based; None = miss
        rank = (
            retrieved_ids.index(item["expected_chunk_id"]) + 1
            if item["expected_chunk_id"] in retrieved_ids
            else None
        )

        for k in K_VALUES:
            if rank is not None and rank <= k:
                recall[k] += 1
        if rank is not None:
            mrr_total += 1.0 / rank
            ndcg_total += 1.0 / math.log2(rank + 1)

    n = len(testset)
    return {
        "strategy": strategy,
        **{f"recall@{k}": recall[k] / n for k in K_VALUES},
        "precision@5": (recall[5] / n) / 5,  # single-relevant simplification
        "mrr": mrr_total / n,
        "ndcg@10": ndcg_total / n,
        "hit_rate@10": recall[10] / n,
        "avg_latency_ms": 1000 * sum(latencies) / n,
    }


def main() -> None:
    with open(PROJECT_ROOT / "eval" / "testset.json") as f:
        testset = json.load(f)
    print(f"{len(testset)} test questions\n")

    pipeline = Pipeline(PROJECT_ROOT / "data" / "processed")

    rows = []
    for strategy in STRATEGIES:
        print(f"evaluating {strategy} ...")
        row = evaluate_strategy(pipeline, testset, strategy)
        rows.append(row)

    columns = list(rows[0].keys())

    # --- markdown report ---
    lines = [
        "# Retrieval Evaluation Results",
        "",
        f"{len(testset)} synthetic questions, single relevant chunk each, retrieve k={RETRIEVE_K}.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        cells = [
            row["strategy"]
            if col == "strategy"
            else (f"{row[col]:.1f}" if col == "avg_latency_ms" else f"{row[col]:.3f}")
            for col in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (PROJECT_ROOT / "eval" / "results.md").write_text("\n".join(lines) + "\n")

    # --- csv, for charts ---
    with open(PROJECT_ROOT / "eval" / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "\n".join(lines[4:]))
    print("\nwrote eval/results.md and eval/results.csv")


if __name__ == "__main__":
    main()
