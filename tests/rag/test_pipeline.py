"""Tests for :mod:`rag.pipeline` and :mod:`rag.evaluation`.

These tests use a hand-rolled retriever and a fake LLM so they do not touch
HuggingFace or any provider API.
"""
from __future__ import annotations

from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag.evaluation import EvalExample, evaluate_pipeline, evaluate_retriever
from rag.pipeline import RAGPipeline, format_context
from rag.retrievers.base import BaseRetriever, RetrievalResult


class _ScriptedRetriever(BaseRetriever):
    """Returns a fixed list of documents regardless of query."""

    def __init__(self, documents: List[Document]) -> None:
        self._documents = list(documents)

    def add_documents(self, documents):
        self._documents.extend(documents)

    def retrieve(self, query: str, k: int = 5):
        return [RetrievalResult(doc, 0.0) for doc in self._documents[:k]]


def test_format_context_includes_metadata():
    docs = [Document(page_content="hello", metadata={"source": "f.pdf", "page": 1})]
    text = format_context(docs)
    assert "source=f.pdf" in text
    assert "page=1" in text
    assert "hello" in text


def test_pipeline_answer_returns_response():
    docs = [Document(page_content="Apple reported $400B revenue.", metadata={"doc_id": "a"})]
    retriever = _ScriptedRetriever(docs)
    fake_llm = FakeListChatModel(responses=["Apple's revenue is $400B."])

    pipeline = RAGPipeline(retriever=retriever, llm=fake_llm, top_k=1)
    response = pipeline.answer("What was Apple's revenue?")

    assert response.question == "What was Apple's revenue?"
    assert response.answer == "Apple's revenue is $400B."
    assert response.contexts[0].metadata["doc_id"] == "a"


def test_pipeline_rejects_non_positive_top_k():
    import pytest

    fake_llm = FakeListChatModel(responses=["x"])
    with pytest.raises(ValueError):
        RAGPipeline(retriever=_ScriptedRetriever([]), llm=fake_llm, top_k=0)


def test_evaluate_retriever_metrics():
    docs = [
        Document(page_content="alpha", metadata={"doc_id": "a"}),
        Document(page_content="beta", metadata={"doc_id": "b"}),
    ]
    retriever = _ScriptedRetriever(docs)

    dataset = [
        EvalExample(question="q1", relevant_doc_ids=["a"]),
        EvalExample(question="q2", relevant_doc_ids=["b"]),
    ]
    report = evaluate_retriever(retriever, dataset, k=2)

    assert report.metrics["hit@2"] == 1.0
    assert 0 < report.metrics["mrr"] <= 1.0
    assert "map" in report.metrics


def test_evaluate_pipeline_skips_generation_metrics_without_reference():
    docs = [Document(page_content="alpha", metadata={"doc_id": "a"})]
    retriever = _ScriptedRetriever(docs)
    fake_llm = FakeListChatModel(responses=["fine"])

    pipeline = RAGPipeline(retriever=retriever, llm=fake_llm, top_k=1)
    dataset = [EvalExample(question="q", relevant_doc_ids=["a"])]
    report = evaluate_pipeline(pipeline, dataset, k=1, include_generation=False)

    assert "answer_similarity" not in report.per_example[0]
    assert report.metrics["hit@1"] == 1.0


def test_pipeline_accepts_plain_callable_llm():
    docs = [Document(page_content="Tesla shipped 1.8M cars.", metadata={"doc_id": "t"})]
    retriever = _ScriptedRetriever(docs)

    def my_llm(prompt: str) -> str:
        assert "Tesla shipped 1.8M cars." in prompt
        return "Tesla shipped 1.8 million cars."

    pipeline = RAGPipeline(retriever=retriever, llm=my_llm, top_k=1)
    response = pipeline.answer("How many cars did Tesla ship?")

    assert response.answer == "Tesla shipped 1.8 million cars."
    assert response.contexts[0].metadata["doc_id"] == "t"


def test_pipeline_accepts_lambda_llm():
    docs = [Document(page_content="alpha", metadata={"doc_id": "a"})]
    retriever = _ScriptedRetriever(docs)

    pipeline = RAGPipeline(
        retriever=retriever,
        llm=lambda prompt: "lambda-answer",
        top_k=1,
    )
    response = pipeline.answer("q")
    assert response.answer == "lambda-answer"


def test_evaluate_pipeline_includes_generation_metrics(monkeypatch):
    docs = [Document(page_content="Apple revenue was $400B.", metadata={"doc_id": "a"})]
    retriever = _ScriptedRetriever(docs)
    fake_llm = FakeListChatModel(responses=["Apple's revenue was $400B."])

    pipeline = RAGPipeline(retriever=retriever, llm=fake_llm, top_k=1)

    # Deterministic stand-in for embed_texts: each text gets a vector built
    # from its hash so identical strings stay identical, but no model is loaded.
    def fake_embed_texts(texts, *args, **kwargs):
        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(8).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            vectors.append(v)
        return np.stack(vectors)

    monkeypatch.setattr("rag.metrics.generation.embed_texts", fake_embed_texts)

    dataset = [
        EvalExample(
            question="What was Apple's revenue?",
            relevant_doc_ids=["a"],
            reference_answer="Apple's revenue was $400B.",
        )
    ]
    report = evaluate_pipeline(pipeline, dataset, k=1, include_generation=True)

    record = report.per_example[0]
    assert "answer_similarity" in record
    assert "faithfulness" in record
    assert "context_precision" in record
