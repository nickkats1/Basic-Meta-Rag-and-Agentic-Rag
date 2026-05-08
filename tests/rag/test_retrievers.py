"""Tests for the retriever implementations.

Dense, hybrid, and reranking retrievers are tested with the bi-encoder /
cross-encoder loaders mocked so the suite does not need to download model
weights from HuggingFace.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from langchain_core.documents import Document

from rag.retrievers.base import BaseRetriever, RetrievalResult
from rag.retrievers.bm25_retriever import BM25Retriever, default_tokenize
from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from rag.retrievers.reranker import RerankerRetriever


# ---------- BM25 ----------


class TestBM25Retriever:
    def test_retrieve_finds_keyword_match(self, toy_corpus):
        retriever = BM25Retriever()
        retriever.add_documents(toy_corpus)
        hits = retriever.retrieve("iPhone revenue", k=2)
        assert hits, "expected at least one hit"
        assert hits[0].document.metadata["doc_id"] == "a"
        assert hits[0].score > 0

    def test_retrieve_returns_at_most_k(self, toy_corpus):
        retriever = BM25Retriever()
        retriever.add_documents(toy_corpus)
        assert len(retriever.retrieve("revenue", k=3)) <= 3

    def test_retrieve_before_indexing_raises(self):
        with pytest.raises(RuntimeError):
            BM25Retriever().retrieve("anything")

    def test_default_tokenize(self):
        assert default_tokenize("Hello, World!") == ["hello", "world"]

    def test_len(self, toy_corpus):
        retriever = BM25Retriever()
        retriever.add_documents(toy_corpus)
        assert len(retriever) == len(toy_corpus)


# ---------- Dense ----------


def _fake_embed(texts, **kwargs):
    """Map each unique token to a deterministic basis-aligned vector.

    Two strings that share many tokens end up close in cosine space, so query
    "iPhone Apple" will best match the document with both tokens.
    """
    vocab = {
        "apple": 0,
        "iphone": 1,
        "tesla": 2,
        "vehicles": 3,
        "fed": 4,
        "interest": 5,
        "microsoft": 6,
        "cloud": 7,
        "amazon": 8,
        "aws": 9,
        "google": 10,
        "advertising": 11,
        "revenue": 12,
        "rates": 13,
    }
    vectors = []
    for text in texts:
        vec = np.zeros(len(vocab), dtype=np.float32)
        for token in text.lower().split():
            stripped = token.strip(".,;:'\"!?")
            if stripped in vocab:
                vec[vocab[stripped]] = 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        vectors.append(vec)
    return np.vstack(vectors)


class TestDenseRetriever:
    def test_retrieve_finds_semantic_match(self, toy_corpus):
        retriever = DenseRetriever()
        with patch("rag.retrievers.dense_retriever.embed_texts", side_effect=_fake_embed):
            retriever.add_documents(toy_corpus)
            hits = retriever.retrieve("apple iphone revenue", k=2)
        assert hits[0].document.metadata["doc_id"] == "a"

    def test_retrieve_before_indexing_raises(self):
        with pytest.raises(RuntimeError):
            DenseRetriever().retrieve("anything")

    def test_add_documents_empty_raises(self):
        with pytest.raises(ValueError):
            DenseRetriever().add_documents([])

    def test_save_and_load_roundtrip(self, tmp_path, toy_corpus):
        original = DenseRetriever()
        with patch("rag.retrievers.dense_retriever.embed_texts", side_effect=_fake_embed):
            original.add_documents(toy_corpus)
            original.save(tmp_path / "index")

            restored = DenseRetriever.load(tmp_path / "index")
            assert len(restored) == len(toy_corpus)
            hits = restored.retrieve("apple iphone revenue", k=2)
        assert hits[0].document.metadata["doc_id"] == "a"

    def test_save_empty_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            DenseRetriever().save(tmp_path / "empty")

    def test_load_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DenseRetriever.load(tmp_path / "nope")


# ---------- Hybrid ----------


class TestRRF:
    def test_doc_in_both_lists_outranks_singleton(self):
        doc_a = Document(page_content="A")
        doc_b = Document(page_content="B")
        doc_c = Document(page_content="C")
        list1 = [
            RetrievalResult(doc_a, 0.0),
            RetrievalResult(doc_b, 0.0),
        ]
        list2 = [
            RetrievalResult(doc_c, 0.0),
            RetrievalResult(doc_a, 0.0),
        ]
        fused = reciprocal_rank_fusion([list1, list2], rank_constant=60)
        # ``doc_a`` appears in both lists -> highest fused score.
        assert fused[0][0].page_content == "A"

    def test_weights_change_winner(self):
        doc_a = Document(page_content="A")
        doc_b = Document(page_content="B")
        list1 = [RetrievalResult(doc_a, 0.0)]  # ranked top by retriever 1 only
        list2 = [RetrievalResult(doc_b, 0.0)]  # ranked top by retriever 2 only

        # Equal weights -> tie broken by insertion order (A wins).
        fused_equal = reciprocal_rank_fusion([list1, list2])
        assert {fused_equal[0][0].page_content, fused_equal[1][0].page_content} == {"A", "B"}

        # Heavy weight on the second list flips the winner to B.
        fused_skewed = reciprocal_rank_fusion(
            [list1, list2], weights=[1.0, 10.0]
        )
        assert fused_skewed[0][0].page_content == "B"

    def test_mismatched_weights_raise(self):
        doc_a = Document(page_content="A")
        with pytest.raises(ValueError):
            reciprocal_rank_fusion(
                [[RetrievalResult(doc_a, 0.0)]],
                weights=[1.0, 2.0],
            )


class TestHybridRetriever:
    def test_indexes_in_both_components(self, toy_corpus):
        hybrid = HybridRetriever()
        with patch("rag.retrievers.dense_retriever.embed_texts", side_effect=_fake_embed):
            hybrid.add_documents(toy_corpus)
            hits = hybrid.retrieve("apple iphone", k=2)
        assert hits, "expected fused hits"
        assert hits[0].document.metadata["doc_id"] == "a"


# ---------- Reranker ----------


class _FakeBase(BaseRetriever):
    """Minimal in-memory base retriever for the reranker test."""

    def __init__(self, documents):
        self._documents = list(documents)

    def add_documents(self, documents):
        self._documents.extend(documents)

    def retrieve(self, query, k=5):
        return [RetrievalResult(doc, 0.0) for doc in self._documents[:k]]


class _FakeCrossEncoder:
    """Predict cross-encoder scores from substring presence."""

    def __init__(self, target):
        self._target = target

    def predict(self, pairs, batch_size=32):  # noqa: ARG002 - mimics real signature
        return np.array(
            [1.0 if self._target in doc.lower() else 0.1 for _, doc in pairs],
            dtype=np.float32,
        )


class TestRerankerRetriever:
    def test_reranks_to_target(self, toy_corpus):
        base = _FakeBase(toy_corpus)
        reranker = RerankerRetriever(base, candidate_pool=len(toy_corpus))
        with patch(
            "rag.retrievers.reranker.load_cross_encoder",
            return_value=_FakeCrossEncoder("microsoft"),
        ):
            hits = reranker.retrieve("cloud growth", k=1)
        assert hits[0].document.metadata["doc_id"] == "d"

    def test_empty_base_returns_empty(self):
        reranker = RerankerRetriever(_FakeBase([]))
        assert reranker.retrieve("anything", k=5) == []
