"""Retriever implementations.

Four retrieval strategies share the :class:`BaseRetriever` interface:

* :class:`BM25Retriever` -- pure sparse keyword retrieval (``rank_bm25``).
* :class:`DenseRetriever` -- bi-encoder embeddings indexed in FAISS.
* :class:`HybridRetriever` -- BM25 + dense fused with Reciprocal Rank Fusion.
* :class:`RerankerRetriever` -- cross-encoder reranking on top of any base
  retriever.
"""
from __future__ import annotations

from rag.retrievers.base import BaseRetriever, RetrievalResult
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever
from rag.retrievers.hybrid_retriever import HybridRetriever
from rag.retrievers.reranker import RerankerRetriever

__all__ = [
    "BM25Retriever",
    "BaseRetriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankerRetriever",
    "RetrievalResult",
]
