"""Cross-encoder reranking on top of any base retriever.

A cross-encoder jointly encodes ``(query, document)`` pairs and scores them
with full attention across both -- much more accurate than bi-encoder cosine
similarity, but quadratic in pair count, so it is only practical as a second
stage that reranks a small candidate set produced by a fast first stage.
"""
from __future__ import annotations

import logging
from typing import List, Sequence

from langchain_core.documents import Document

from rag.embeddings import DEFAULT_CROSS_ENCODER, load_cross_encoder
from rag.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class RerankerRetriever(BaseRetriever):
    """Wraps a base retriever and reranks its hits with a cross-encoder.

    The wrapped retriever produces ``candidate_pool`` first-stage hits per
    query; the cross-encoder rescores them and the top ``k`` are returned.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_name: str = DEFAULT_CROSS_ENCODER,
        candidate_pool: int = 20,
        batch_size: int = 32,
    ) -> None:
        """Build a reranking retriever.

        Args:
            base_retriever: First-stage retriever (any :class:`BaseRetriever`).
            model_name: HuggingFace cross-encoder id.
            candidate_pool: How many first-stage hits to fetch before
                reranking. Should be >> the requested ``k``.
            batch_size: Cross-encoder batch size at scoring time.
        """
        self._base = base_retriever
        self._model_name = model_name
        self._candidate_pool = candidate_pool
        self._batch_size = batch_size

    @property
    def base_retriever(self) -> BaseRetriever:
        """The wrapped first-stage retriever."""
        return self._base

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Forward to the base retriever's index."""
        self._base.add_documents(documents)

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Retrieve, rerank with the cross-encoder, and return top ``k``.

        Args:
            query: Natural language query.
            k: Final number of hits to return after reranking.

        Returns:
            Up to ``k`` :class:`RetrievalResult` hits sorted by cross-encoder
            relevance score.
        """
        candidates = self._base.retrieve(query, k=self._candidate_pool)
        if not candidates:
            return []

        cross_encoder = load_cross_encoder(self._model_name)
        pairs = [(query, hit.document.page_content) for hit in candidates]
        scores = cross_encoder.predict(pairs, batch_size=self._batch_size)

        reranked = sorted(
            (
                RetrievalResult(document=hit.document, score=float(score))
                for hit, score in zip(candidates, scores)
            ),
            key=lambda r: r.score,
            reverse=True,
        )
        return reranked[:k]
