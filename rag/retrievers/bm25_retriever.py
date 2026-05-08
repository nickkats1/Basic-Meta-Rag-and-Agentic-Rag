"""Pure BM25 (sparse keyword) retriever using ``rank_bm25``.

BM25 is a strong baseline for keyword-heavy queries (e.g. exact ticker symbols,
defined terms, dollar figures) and serves as the sparse half of the hybrid
retriever in :mod:`rag.retrievers.hybrid_retriever`.
"""
from __future__ import annotations

import logging
import re
from typing import List, Sequence

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def default_tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word characters.

    A deliberately small tokenizer: lowercased ``\\w+`` matches. Good enough for
    most English-language SEC text and avoids dragging in heavy NLP deps.

    Args:
        text: Raw text to tokenize.

    Returns:
        A list of lowercased token strings.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """Sparse keyword retriever backed by ``rank_bm25.BM25Okapi``.

    Documents are tokenized once at index time. ``retrieve`` re-tokenizes the
    query and asks the BM25 index for the top ``k`` document scores.

    Example:
        >>> retriever = BM25Retriever()
        >>> retriever.add_documents([Document(page_content="Revenue rose 12%")])
        >>> hits = retriever.retrieve("revenue", k=1)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize an empty BM25 index.

        Args:
            k1: BM25 term-frequency saturation parameter.
            b: BM25 length-normalization parameter.
        """
        self._k1 = k1
        self._b = b
        self._documents: List[Document] = []
        self._tokenized: List[List[str]] = []
        self._bm25 = None

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index ``documents``. Re-fits the BM25 index on every call.

        Args:
            documents: Documents to add. Must be non-empty.

        Raises:
            ValueError: If ``documents`` is empty.
        """
        if not documents:
            raise ValueError("documents cannot be empty")

        self._documents.extend(documents)
        self._tokenized.extend(default_tokenize(doc.page_content) for doc in documents)
        self._bm25 = BM25Okapi(self._tokenized, k1=self._k1, b=self._b)
        logger.info("BM25 index now contains %d documents", len(self._documents))

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Return the top ``k`` BM25 hits for ``query``.

        Args:
            query: Natural language query.
            k: Number of hits to return.

        Returns:
            Up to ``k`` :class:`RetrievalResult` objects sorted by BM25 score.

        Raises:
            RuntimeError: If called before any documents have been added.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever has no indexed documents; call add_documents first")

        tokenized_query = default_tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        top_k = min(k, len(self._documents))
        # ``argsort`` is ascending; take the last ``top_k`` and reverse.
        top_indices = scores.argsort()[-top_k:][::-1]
        return [
            RetrievalResult(document=self._documents[i], score=float(scores[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._documents)
