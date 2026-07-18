"""Tests for CLI module."""

from __future__ import annotations

import pytest

from cli import get_retriever
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)


class TestGetRetriever:
    """Test retriever factory function."""

    def test_get_retriever_bm25(self):
        """get_retriever returns BM25Retriever for bm25 type."""
        retriever = get_retriever("bm25")
        assert isinstance(retriever, BM25Retriever)

    def test_get_retriever_dense(self):
        """get_retriever returns DenseRetriever for dense type."""
        retriever = get_retriever("dense")
        assert isinstance(retriever, DenseRetriever)

    def test_get_retriever_hybrid(self):
        """get_retriever returns HybridRetriever for hybrid type."""
        retriever = get_retriever("hybrid")
        assert isinstance(retriever, HybridRetriever)

    def test_get_retriever_reranker(self):
        """get_retriever returns RerankerRetriever for reranker type."""
        retriever = get_retriever("reranker")
        assert isinstance(retriever, RerankerRetriever)

    def test_get_retriever_invalid_type(self):
        """get_retriever raises ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown retriever"):
            get_retriever("unknown")
