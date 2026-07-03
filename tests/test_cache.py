"""Semantic cache tests — paraphrase hits, unrelated misses."""

from src.rag.cache import SemanticCache


def test_exact_repeat_is_a_hit():
    cache = SemanticCache()
    cache.put("does creatine cause cramps?", {"answer": "cached"})
    hit = cache.get("does creatine cause cramps?")
    assert hit is not None and hit[0]["answer"] == "cached"
    assert hit[1] >= 0.99  # identical text → ~1.0 similarity


def test_paraphrase_is_a_hit():
    cache = SemanticCache(threshold=0.7)
    cache.put("does creatine cause muscle cramps?", {"answer": "cached"})
    hit = cache.get("can creatine give me cramps in my muscles?")
    assert hit is not None, "a close paraphrase should hit the cache"


def test_unrelated_question_is_a_miss():
    cache = SemanticCache(threshold=0.9)
    cache.put("does creatine cause cramps?", {"answer": "cached"})
    assert cache.get("what is reciprocal rank fusion?") is None


def test_empty_cache_returns_none():
    assert SemanticCache().get("anything") is None
