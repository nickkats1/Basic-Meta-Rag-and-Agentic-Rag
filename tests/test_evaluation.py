"""Tests for RAG evaluation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rag.data_ingestion import Document
from rag.evaluation import (
    EvalExample,
    EvalResult,
    evaluate_pipeline,
    evaluate_retriever,
)
from rag.pipeline import RAGPipeline
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
def eval_documents():
    """Create sample documents for evaluation."""
    return [
        Document(
            page_content="Paris is the capital of France",
            metadata={"doc_id": "chunk_0", "source": "test.pdf"},
        ),
        Document(
            page_content="France is in Western Europe",
            metadata={"doc_id": "chunk_1", "source": "test.pdf"},
        ),
        Document(
            page_content="The Eiffel Tower is in Paris",
            metadata={"doc_id": "chunk_2", "source": "test.pdf"},
        ),
    ]


@pytest.fixture
def eval_examples():
    """Create sample evaluation examples."""
    return [
        EvalExample(
            question="What is the capital of France?",
            relevant_doc_ids=["chunk_0", "chunk_1"],
            reference_answer="Paris is the capital of France",
        ),
        EvalExample(
            question="Where is the Eiffel Tower?",
            relevant_doc_ids=["chunk_2"],
            reference_answer="The Eiffel Tower is in Paris",
        ),
    ]


class TestEvalExample:
    """Test EvalExample dataclass."""

    def test_eval_example_creation(self):
        """EvalExample can be created with required fields."""
        example = EvalExample(
            question="Test question?",
            relevant_doc_ids=["chunk_0"],
        )
        assert example.question == "Test question?"
        assert example.relevant_doc_ids == ["chunk_0"]
        assert example.reference_answer == ""

    def test_eval_example_with_reference(self):
        """EvalExample can include reference answer."""
        example = EvalExample(
            question="Test question?",
            relevant_doc_ids=["chunk_0"],
            reference_answer="Reference answer",
        )
        assert example.reference_answer == "Reference answer"


class TestEvalResult:
    """Test EvalResult dataclass."""

    def test_eval_result_creation(self):
        """EvalResult can be created with default values."""
        result = EvalResult()
        assert result.metrics == {}
        assert result.per_example == []

    def test_eval_result_with_metrics(self):
        """EvalResult can be created with metrics."""
        metrics = {"hit@5": 0.8, "precision@5": 0.9}
        result = EvalResult(metrics=metrics)
        assert result.metrics == metrics


class TestEvaluateRetriever:
    """Test retriever evaluation."""

    def test_evaluate_retriever_returns_result(self, eval_documents, eval_examples):
        """evaluate_retriever returns EvalResult."""
        retriever = DummyRetriever(documents=eval_documents)
        result = evaluate_retriever(retriever, eval_examples, k=5)

        assert isinstance(result, EvalResult)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.per_example, list)

    def test_evaluate_retriever_computes_metrics(
        self, eval_documents, eval_examples
    ):
        """evaluate_retriever computes retrieval metrics."""
        retriever = DummyRetriever(documents=eval_documents)
        result = evaluate_retriever(retriever, eval_examples, k=5)

        assert "hit@5" in result.metrics
        assert "precision@5" in result.metrics
        assert "recall@5" in result.metrics
        assert "mrr" in result.metrics
        assert "ndcg@5" in result.metrics
        assert "map" in result.metrics

    def test_evaluate_retriever_per_example_metrics(
        self, eval_documents, eval_examples
    ):
        """evaluate_retriever computes per-example metrics."""
        retriever = DummyRetriever(documents=eval_documents)
        result = evaluate_retriever(retriever, eval_examples, k=5)

        assert len(result.per_example) == 2
        for example_result in result.per_example:
            assert "hit" in example_result
            assert "precision" in example_result
            assert "recall" in example_result
            assert "mrr" in example_result
            assert "ndcg" in example_result
            assert "ap" in example_result


class TestEvaluatePipeline:
    """Test pipeline evaluation."""

    def test_evaluate_pipeline_returns_result(
        self, eval_documents, eval_examples
    ):
        """evaluate_pipeline returns EvalResult."""
        retriever = DummyRetriever(documents=eval_documents)
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Test answer"
        pipeline = RAGPipeline(retriever=retriever, llm=mock_llm, top_k=5)

        result = evaluate_pipeline(pipeline, eval_examples, k=5)

        assert isinstance(result, EvalResult)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.per_example, list)

    def test_evaluate_pipeline_without_reference_answers(
        self, eval_documents
    ):
        """evaluate_pipeline works without reference answers."""
        examples = [
            EvalExample(
                question="Test question?",
                relevant_doc_ids=["chunk_0"],
            )
        ]
        retriever = DummyRetriever(documents=eval_documents)
        mock_llm = MagicMock()
        pipeline = RAGPipeline(retriever=retriever, llm=mock_llm, top_k=5)

        result = evaluate_pipeline(pipeline, examples, k=5)

        assert isinstance(result, EvalResult)
        assert "hit@5" in result.metrics
