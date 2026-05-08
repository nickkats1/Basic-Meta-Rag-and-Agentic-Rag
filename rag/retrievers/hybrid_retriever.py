"""Hybrid sparse+dense retriever using (weighted) Reciprocal Rank Fusion.

Reciprocal Rank Fusion (RRF) is a parameter-light fusion method that combines
multiple ranked lists by summing ``weight / (rank_constant + rank_i)`` across
retrievers. It is robust to score-scale differences, which makes it a natural
fit for combining BM25 (unbounded, document-length-sensitive) with cosine
similarity (bounded in ``[-1, 1]``).

Per-list weights let callers tilt the fusion toward sparse or dense matches
when a domain has a clear preference.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from rag.retrievers.base import BaseRetriever, RetrievalResult
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


def _dedup_key(document: Document) -> str:
    """Return a stable identifier for ``document`` for fusion deduplication.

    Prefers ``metadata['doc_id']`` (assigned by
    :func:`rag.data_ingestion.chunk_documents`) so that two distinct chunks
    with identical text don't collapse, falling back to ``page_content`` when
    no id is set.
    """
    if document.metadata and "doc_id" in document.metadata:
        return f"id:{document.metadata['doc_id']}"
    return f"text:{document.page_content}"


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    rank_constant: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[Tuple[Document, float]]:
    """Fuse multiple ranked lists with (weighted) Reciprocal Rank Fusion.

    Documents are de-duplicated by ``metadata['doc_id']`` when present,
    falling back to ``page_content``. The score for a document is
    ``sum_i weight_i / (rank_constant + rank_i)`` across all ranked lists
    that contain it, where ``rank_i`` is the 0-indexed position of the
    document in list ``i``.

    Args:
        ranked_lists: One or more ranked lists of :class:`RetrievalResult`.
        rank_constant: RRF smoothing constant. Cormack et al. (2009) recommend
            60.
        weights: Optional per-list weights. When omitted every list weighs
            equally. Must have the same length as ``ranked_lists``.

    Returns:
        List of ``(document, fused_score)`` tuples sorted by descending score.

    Raises:
        ValueError: If ``weights`` is supplied with a different length than
            ``ranked_lists``.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    fused: Dict[str, float] = defaultdict(float)
    representatives: Dict[str, Document] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, hit in enumerate(ranked):
            key = _dedup_key(hit.document)
            fused[key] += weight / (rank_constant + rank)
            representatives.setdefault(key, hit.document)

    return sorted(
        ((representatives[k], score) for k, score in fused.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


class HybridRetriever(BaseRetriever):
    """BM25 + dense bi-encoder retriever fused via weighted RRF."""

    def __init__(
        self,
        bm25: Optional[BM25Retriever] = None,
        dense: Optional[DenseRetriever] = None,
        rank_constant: int = 60,
        candidates_per_retriever: int = 20,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            bm25: Optional pre-built BM25 retriever. A fresh one is created if
                omitted.
            dense: Optional pre-built dense retriever. A fresh one is created
                if omitted.
            rank_constant: RRF smoothing constant.
            candidates_per_retriever: Number of candidates pulled from each
                base retriever before fusion. Should typically be larger than
                the requested ``k`` to give RRF room to re-rank.
            bm25_weight: Multiplicative weight applied to BM25 contributions.
            dense_weight: Multiplicative weight applied to dense contributions.

        Raises:
            ValueError: If either weight is non-positive.
        """
        if bm25_weight <= 0 or dense_weight <= 0:
            raise ValueError("Weights must be positive")

        self._bm25 = bm25 or BM25Retriever()
        self._dense = dense or DenseRetriever()
        self._rank_constant = rank_constant
        self._candidates = candidates_per_retriever
        self._bm25_weight = bm25_weight
        self._dense_weight = dense_weight

    @property
    def bm25(self) -> BM25Retriever:
        """The underlying BM25 retriever."""
        return self._bm25

    @property
    def dense(self) -> DenseRetriever:
        """The underlying dense retriever."""
        return self._dense

    @property
    def weights(self) -> Tuple[float, float]:
        """The ``(bm25_weight, dense_weight)`` pair."""
        return (self._bm25_weight, self._dense_weight)

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index ``documents`` in both component retrievers."""
        self._bm25.add_documents(documents)
        self._dense.add_documents(documents)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve and fuse hits from both component retrievers.

        Args:
            query: Natural language query.
            k: Number of fused hits to return.

        Returns:
            Up to ``k`` :class:`RetrievalResult` hits with weighted RRF scores.
        """
        bm25_hits = self._bm25.retrieve(query, k=self._candidates)
        dense_hits = self._dense.retrieve(query, k=self._candidates)
        fused = reciprocal_rank_fusion(
            [bm25_hits, dense_hits],
            rank_constant=self._rank_constant,
            weights=[self._bm25_weight, self._dense_weight],
        )
        return [RetrievalResult(document=doc, score=score) for doc, score in fused[:k]]
