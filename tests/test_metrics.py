"""Tests for metrics."""

from __future__ import annotations


from rag.metrics import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRetrievalMetrics:
    """Test retrieval metrics."""

    def test_hit_rate_at_k(self):
        assert hit_rate_at_k(["a", "b", "c"], ["a"], k=5) == 1.0
        assert hit_rate_at_k(["x", "y", "z"], ["a"], k=5) == 0.0
        assert hit_rate_at_k(["a", "b"], ["c", "d"], k=2) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b"], k=2) == 1.0
        assert precision_at_k(["a", "x", "y"], ["a"], k=3) == 1.0 / 3.0
        assert precision_at_k(["x", "y"], ["a"], k=2) == 0.0

    def test_recall_at_k(self):
        assert recall_at_k(["a", "b"], ["a", "b", "c"], k=2) == 2.0 / 3.0
        assert recall_at_k(["a"], ["a", "b"], k=1) == 0.5
        assert recall_at_k(["x"], ["a"], k=1) == 0.0

    def test_mean_reciprocal_rank(self):
        assert mean_reciprocal_rank(["a", "b"], ["a"]) == 1.0
        assert mean_reciprocal_rank(["x", "a", "b"], ["a"]) == 0.5
        assert mean_reciprocal_rank(["x"], ["a"]) == 0.0

    def test_average_precision(self):
        result = average_precision(["a", "b", "c"], ["a", "b"])
        assert result > 0.0

    def test_mean_average_precision(self):
        all_retrieved = [["a", "b"], ["x", "y"]]
        all_relevant = [["a"], ["x"]]
        result = mean_average_precision(all_retrieved, all_relevant)
        assert result > 0.0

    def test_ndcg_at_k(self):
        result = ndcg_at_k(["a", "b", "c"], ["a", "b"], k=3)
        assert 0.0 <= result <= 1.0
