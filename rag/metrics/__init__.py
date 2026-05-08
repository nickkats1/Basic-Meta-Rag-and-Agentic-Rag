"""RAG evaluation metrics.

Two flavors are exposed:

* :mod:`rag.metrics.retrieval` -- order-aware retrieval metrics (Hit@k,
  Recall@k, Precision@k, MRR, NDCG@k).
* :mod:`rag.metrics.generation` -- embedding-based answer-quality metrics
  (semantic answer similarity, embedding-based faithfulness).
"""
from __future__ import annotations

from rag.metrics.generation import (
    answer_similarity,
    context_precision,
    embedding_faithfulness,
)
from rag.metrics.retrieval import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "answer_similarity",
    "average_precision",
    "context_precision",
    "embedding_faithfulness",
    "hit_rate_at_k",
    "mean_average_precision",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
