"""Sec-Rag: a multi-strategy Retrieval-Augmented Generation toolkit.

Public surface:

* :mod:`rag.data_ingestion` -- PDF loading and chunking.
* :mod:`rag.embeddings` -- HuggingFace bi-encoder / cross-encoder loaders.
* :mod:`rag.retrievers` -- BM25, dense, hybrid, and reranking retrievers.
* :mod:`rag.metrics` -- retrieval and generation metrics.
* :mod:`rag.pipeline` -- end-to-end RAG pipeline.
* :mod:`rag.evaluation` -- evaluation harness for retrievers and pipelines.
* :mod:`rag.llm` -- multi-provider LLM factory.
"""
from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
