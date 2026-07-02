# Retrieval Evaluation Results

40 synthetic questions, single relevant chunk each, retrieve k=10.

| strategy | recall@1 | recall@5 | recall@10 | precision@5 | mrr | ndcg@10 | hit_rate@10 | avg_latency_ms |
|---|---|---|---|---|---|---|---|---|
| dense | 0.300 | 0.625 | 0.675 | 0.125 | 0.409 | 0.473 | 0.675 | 250.8 |
| bm25 | 0.525 | 0.825 | 0.925 | 0.165 | 0.643 | 0.711 | 0.925 | 0.5 |
| hybrid | 0.425 | 0.825 | 0.850 | 0.165 | 0.597 | 0.661 | 0.850 | 8.7 |
| rerank | 0.300 | 0.600 | 0.950 | 0.120 | 0.468 | 0.580 | 0.950 | 1956.1 |
