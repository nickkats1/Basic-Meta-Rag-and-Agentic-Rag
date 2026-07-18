"""Tests for RAG pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.data_ingestion import Document
from rag.pipeline import RAGPipeline, RAGResult
from rag.retrievers import Retriever


class DummyRetriever(Retriever):
    """A simple retriever for testing."""

    def __init__(self, documents: list[Document] | None = None):
        self.documents = documents or []

    def add_documents(self, documents: list[Document]) -> None:
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.documents[:top_k]


@pytest.fixture
def dummy_retriever():
    """Create a dummy retriever with sample documents."""
    docs = [
        Document(
            page_content="The quick brown fox",
            metadata={"doc_id": "chunk_0", "source": "test.pdf"},
        ),
        Document(
            page_content="jumps over the lazy dog",
            metadata={"doc_id": "chunk_1", "source": "test.pdf"},
        ),
    ]
    return DummyRetriever(documents=docs)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    llm = MagicMock()
    llm.generate.return_value = "The fox jumps over the dog."
    return llm


class TestRAGPipeline:
    """Test RAG pipeline."""

    def test_rag_result_creation(self):
        """RAGResult can be created with answer, contexts, and doc_ids."""
        result = RAGResult(
            answer="Test answer",
            contexts=["Context 1", "Context 2"],
            doc_ids=["chunk_0", "chunk_1"],
        )
        assert result.answer == "Test answer"
        assert len(result.contexts) == 2
        assert len(result.doc_ids) == 2

    def test_pipeline_initialization(self, dummy_retriever, mock_llm):
        """RAGPipeline initializes with retriever and LLM."""
        pipeline = RAGPipeline(
            retriever=dummy_retriever,
            llm=mock_llm,
            top_k=5,
        )
        assert pipeline.retriever == dummy_retriever
        assert pipeline.llm == mock_llm
        assert pipeline.top_k == 5

    def test_pipeline_answer(self, dummy_retriever, mock_llm):
        """RAGPipeline.answer returns a RAGResult."""
        pipeline = RAGPipeline(
            retriever=dummy_retriever,
            llm=mock_llm,
            top_k=5,
        )
        result = pipeline.answer("What does the fox do?")

        assert isinstance(result, RAGResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.contexts, list)
        assert isinstance(result.doc_ids, list)

    def test_pipeline_answer_extracts_contexts(self, dummy_retriever, mock_llm):
        """RAGPipeline.answer extracts contexts from retrieved documents."""
        pipeline = RAGPipeline(
            retriever=dummy_retriever,
            llm=mock_llm,
            top_k=5,
        )
        result = pipeline.answer("Test query")

        assert len(result.contexts) == 2
        assert "quick brown fox" in result.contexts[0]
        assert "jumps over the lazy dog" in result.contexts[1]

    def test_pipeline_answer_extracts_doc_ids(self, dummy_retriever, mock_llm):
        """RAGPipeline.answer extracts doc_ids from retrieved documents."""
        pipeline = RAGPipeline(
            retriever=dummy_retriever,
            llm=mock_llm,
            top_k=5,
        )
        result = pipeline.answer("Test query")

        assert len(result.doc_ids) == 2
        assert "chunk_0" in result.doc_ids
        assert "chunk_1" in result.doc_ids
