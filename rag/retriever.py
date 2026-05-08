"""Backwards-compatible thin wrapper around :class:`DenseRetriever`.

Kept so that older code (notebooks, downstream scripts) that imports
``rag.retriever.get_retriever`` continues to work. New code should depend on
:mod:`rag.retrievers` directly.
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.documents import Document

from rag.retrievers.dense_retriever import DenseRetriever


def get_retriever(
    documents: List[Document],
    k: int = 5,
    search_type: str = "similarity",
) -> Any:
    """Build a dense retriever and return a ``retrieve(query, k)`` shim.

    The returned object exposes a LangChain-compatible ``invoke(query)`` method
    so it can be used as a drop-in replacement for the previous
    ``FAISS.as_retriever()`` return value.

    Args:
        documents: Documents to index.
        k: Default number of hits to return.
        search_type: Accepted for backwards compatibility but unused; the
            underlying :class:`DenseRetriever` always uses cosine similarity.

    Returns:
        A small adapter object with an ``invoke(query)`` method that returns a
        list of ``Document`` objects.

    Raises:
        ValueError: If ``documents`` is empty.
    """
    if not documents:
        raise ValueError("must have valid documents")
    del search_type  # accepted for compatibility, FAISS-IP is the only mode

    backend = DenseRetriever()
    backend.add_documents(documents)

    class _CompatRetriever:
        """LangChain-style retriever shim: ``invoke(query) -> List[Document]``."""

        def __init__(self, dense: DenseRetriever, default_k: int) -> None:
            self._dense = dense
            self._k = default_k

        def invoke(self, query: str) -> List[Document]:
            return self._dense.retrieve_documents(query, k=self._k)

        def get_relevant_documents(self, query: str) -> List[Document]:
            return self.invoke(query)

    return _CompatRetriever(backend, k)
