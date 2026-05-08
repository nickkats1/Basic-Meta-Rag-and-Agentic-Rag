"""Tests for :mod:`rag.metrics.retrieval`.

Generation metrics are not unit-tested here because they require loading a
real bi-encoder; they are exercised indirectly via ``test_pipeline``.
"""
from __future__ import annotations

import math

import pytest

from rag.metrics.retrieval import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestHitRate:
    def test_hit_present(self):
        assert hit_rate_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0

    def test_hit_outside_k(self):
        assert hit_rate_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_no_relevant(self):
        assert hit_rate_at_k(["a", "b"], set(), k=2) == 0.0


class TestPrecision:
    def test_two_of_three_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(2 / 3)

    def test_truncation(self):
        assert precision_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=3) == 0.0


class TestRecall:
    def test_full_recall(self):
        assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "b"], {"a", "c"}, k=2) == 0.5

    def test_no_relevant(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


class TestMRR:
    def test_first_position(self):
        assert mean_reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_third_position(self):
        assert mean_reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_match(self):
        assert mean_reciprocal_rank(["x", "y"], {"a"}) == 0.0


class TestAveragePrecision:
    def test_known_value(self):
        # relevant at ranks 1 and 3: AP = (1/1 + 2/3) / 2
        ap = average_precision(["a", "x", "b", "y"], {"a", "b"})
        assert ap == pytest.approx((1.0 + 2 / 3) / 2)

    def test_no_relevant(self):
        assert average_precision(["a", "b"], set()) == 0.0


class TestMAP:
    def test_average_across_queries(self):
        result = mean_average_precision(
            [["a", "b"], ["x", "a"]],
            [{"a"}, {"a"}],
        )
        # AP1 = 1/1 / 1 = 1; AP2 = (1/2)/1 = 0.5; mean = 0.75
        assert result == pytest.approx(0.75)

    def test_misaligned_lengths(self):
        with pytest.raises(ValueError):
            mean_average_precision([["a"]], [{"a"}, {"b"}])


class TestNDCG:
    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_known_partial(self):
        # Single relevant doc at rank 2 -> DCG = 1/log2(3); IDCG = 1/log2(2) = 1
        ndcg = ndcg_at_k(["x", "a", "y"], {"a"}, k=3)
        assert ndcg == pytest.approx(1 / math.log2(3))

    def test_no_relevant(self):
        assert ndcg_at_k(["a"], set(), k=1) == 0.0


def test_invalid_k_rejected():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)
