"""Abstract base class for retrievers.

Every retriever variant in this package conforms to a single, narrow contract
so that they can be composed (e.g. wrapping a :class:`BM25Retriever` with a
:class:`RerankerRetriever`) without leaking implementation details.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalResult:
    """A scored retrieval hit.

    Attributes:
        document: The retrieved ``Document``.
        score: Retriever-specific relevance score. Higher is more relevant.
            Score scales differ across retrievers and should not be compared
            across retriever types directly.
    """

    document: Document
    score: float


class BaseRetriever(ABC):
    """Common interface for all retriever implementations."""

    @abstractmethod
    def add_documents(self, documents: Sequence[Document]) -> None:
        """Index a batch of documents.

        Args:
            documents: Documents to add to the underlying index.
        """

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Return the top ``k`` results for ``query``.

        Args:
            query: Natural language query.
            k: Number of hits to return.

        Returns:
            Up to ``k`` :class:`RetrievalResult` objects, sorted by descending
            score.
        """

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Convenience wrapper that returns only the underlying ``Document`` objects.

        Args:
            query: Natural language query.
            k: Number of hits to return.

        Returns:
            Up to ``k`` ``Document`` objects.
        """
        return [hit.document for hit in self.retrieve(query, k=k)]
