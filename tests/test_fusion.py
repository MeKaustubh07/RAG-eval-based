"""Unit tests for reciprocal rank fusion — the math must be exact."""

from src.rag.hybrid import reciprocal_rank_fusion


def test_agreement_beats_single_list_rank_one():
    # 'c' is rank-2 in both lists; 'a' and 'd' are rank-1 in one list each.
    bm25 = [("a", 9.0), ("c", 7.0), ("b", 6.0)]
    dense = [("d", 0.9), ("c", 0.8), ("e", 0.7)]
    ranked = reciprocal_rank_fusion([bm25, dense])
    assert ranked[0][0] == "c", "a doc ranked well by BOTH systems should win"


def test_scores_are_ignored_only_ranks_matter():
    # Wildly different raw scores, identical ranks → identical fused scores.
    a = reciprocal_rank_fusion([[("x", 1000.0), ("y", 999.0)]])
    b = reciprocal_rank_fusion([[("x", 0.02), ("y", 0.01)]])
    assert [cid for cid, _ in a] == [cid for cid, _ in b]
    assert a[0][1] == b[0][1]


def test_rrf_formula_value():
    # Single list, rank 1 → 1/(60+1).
    ranked = reciprocal_rank_fusion([[("only", 5.0)]], k=60)
    assert abs(ranked[0][1] - 1.0 / 61) < 1e-9


def test_top_n_truncates():
    lst = [(f"d{i}", 1.0 / (i + 1)) for i in range(10)]
    assert len(reciprocal_rank_fusion([lst], top_n=3)) == 3
