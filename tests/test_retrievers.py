"""Tests for retrievers."""

from __future__ import annotations

import pytest

from rag.data_ingestion import Document
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)


@pytest.fixture
def sample_docs():
    """Sample documents for retrieval testing."""
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={"doc_id": "chunk_0", "source": "test.pdf"},
        ),
        Document(
            page_content="A lazy dog sleeps under the big tree all day.",
            metadata={"doc_id": "chunk_1", "source": "test.pdf"},
        ),
        Document(
            page_content="The brown fox is quick and smart.",
            metadata={"doc_id": "chunk_2", "source": "test.pdf"},
        ),
    ]


class TestBM25Retriever:
    """Test BM25 sparse retrieval."""

    def test_retrieves_documents(self, sample_docs):
        retriever = BM25Retriever()
        retriever.add_documents(sample_docs)
        results = retriever.retrieve("fox", top_k=2)
        assert len(results) <= 2
        assert any("fox" in doc.page_content.lower() for doc in results)

    def test_returns_empty_before_indexing(self):
        retriever = BM25Retriever()
        results = retriever.retrieve("query", top_k=5)
        assert results == []


class TestDenseRetriever:
    """Test dense semantic retrieval."""

    def test_retrieves_documents(self, sample_docs):
        retriever = DenseRetriever()
        retriever.add_documents(sample_docs)
        results = retriever.retrieve("quick animal", top_k=2)
        assert len(results) <= 2

    def test_returns_empty_before_indexing(self):
        retriever = DenseRetriever()
        results = retriever.retrieve("query", top_k=5)
        assert results == []


class TestHybridRetriever:
    """Test hybrid BM25 + dense retrieval."""

    def test_retrieves_documents(self, sample_docs):
        retriever = HybridRetriever()
        retriever.add_documents(sample_docs)
        results = retriever.retrieve("fox", top_k=2)
        assert len(results) <= 2

    def test_fuses_results(self, sample_docs):
        retriever = HybridRetriever()
        retriever.add_documents(sample_docs)
        results = retriever.retrieve("brown fox", top_k=3)
        assert len(results) > 0


class TestRerankerRetriever:
    """Test cross-encoder reranking."""

    def test_reranks_results(self, sample_docs):
        base = BM25Retriever()
        retriever = RerankerRetriever(base_retriever=base, candidate_pool=3)
        retriever.add_documents(sample_docs)
        results = retriever.retrieve("fox", top_k=2)
        assert len(results) <= 2
