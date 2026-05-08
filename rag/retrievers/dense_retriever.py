"""Dense retriever backed by a HuggingFace bi-encoder and a FAISS index.

Owns the FAISS index directly (no LangChain ``FAISS`` wrapper) so that
indexing, persistence, and search behavior are transparent and easy to
extend.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Union

import faiss
import numpy as np
from langchain_core.documents import Document

from rag.embeddings import DEFAULT_BI_ENCODER, embed_texts
from rag.retrievers.base import BaseRetriever, RetrievalResult

logger = logging.getLogger(__name__)


_INDEX_FILENAME = "index.faiss"
_DOCS_FILENAME = "documents.pkl"
_META_FILENAME = "meta.json"


class DenseRetriever(BaseRetriever):
    """Bi-encoder + FAISS dense retriever.

    Embeddings are L2-normalized so that the FAISS inner-product index returns
    cosine similarities directly. The index, the document list, and the model
    name can be persisted to disk with :meth:`save` and restored with
    :meth:`load`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER,
        batch_size: int = 32,
    ) -> None:
        """Build an empty dense retriever.

        Args:
            model_name: HuggingFace bi-encoder id.
            batch_size: Encoder batch size used at index time.
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._documents: List[Document] = []
        self._embeddings: Optional[np.ndarray] = None
        self._index = None

    @property
    def model_name(self) -> str:
        """Return the bi-encoder model id."""
        return self._model_name

    def _build_index(self, embeddings: np.ndarray) -> None:
        """Replace the FAISS index with a fresh inner-product index over ``embeddings``."""
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Embed and index ``documents``.

        Args:
            documents: Documents to add to the FAISS index. Must be non-empty.

        Raises:
            ValueError: If ``documents`` is empty.
        """
        if not documents:
            raise ValueError("documents cannot be empty")

        new_embeddings = embed_texts(
            [doc.page_content for doc in documents],
            model_name=self._model_name,
            batch_size=self._batch_size,
            normalize=True,
        )
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        self._documents.extend(documents)
        self._build_index(self._embeddings)
        logger.info("Dense index now contains %d documents", len(self._documents))

    def retrieve(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """Return the top ``k`` cosine-similarity neighbors of ``query``.

        Args:
            query: Natural language query.
            k: Number of hits to return.

        Returns:
            Up to ``k`` :class:`RetrievalResult` hits with cosine-similarity scores.

        Raises:
            RuntimeError: If called before any documents have been added.
        """
        if self._index is None:
            raise RuntimeError("DenseRetriever has no indexed documents; call add_documents first")

        query_emb = embed_texts(
            [query],
            model_name=self._model_name,
            batch_size=self._batch_size,
            normalize=True,
        )
        top_k = min(k, len(self._documents))
        scores, indices = self._index.search(query_emb, top_k)
        return [
            RetrievalResult(document=self._documents[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def save(self, directory: Union[str, Path]) -> None:
        """Persist the FAISS index, documents, and metadata to ``directory``.

        Three files are written: ``index.faiss``, ``documents.pkl``,
        ``meta.json``.

        Args:
            directory: Target directory. Created if it doesn't exist.

        Raises:
            RuntimeError: If the index is empty (nothing to save).
        """
        if self._index is None or self._embeddings is None:
            raise RuntimeError("Nothing to save: DenseRetriever has no indexed documents")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / _INDEX_FILENAME))
        with open(path / _DOCS_FILENAME, "wb") as fh:
            pickle.dump(self._documents, fh, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path / _META_FILENAME, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "model_name": self._model_name,
                    "batch_size": self._batch_size,
                    "n_documents": len(self._documents),
                    "embedding_dim": int(self._embeddings.shape[1]),
                },
                fh,
                indent=2,
            )
        logger.info("Saved DenseRetriever (%d docs) to %s", len(self._documents), path)

    def _restore_from(
        self,
        documents: List[Document],
        index,
        embeddings: np.ndarray,
    ) -> None:
        """Populate state from a previously persisted index. Used by :meth:`load`."""
        self._documents = list(documents)
        self._index = index
        self._embeddings = embeddings

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "DenseRetriever":
        """Restore a retriever previously written by :meth:`save`.

        Args:
            directory: Directory containing ``index.faiss``, ``documents.pkl``,
                and ``meta.json``.

        Returns:
            A fully populated :class:`DenseRetriever`.

        Raises:
            FileNotFoundError: If any of the three expected files is missing.
        """
        path = Path(directory)
        for required in (_INDEX_FILENAME, _DOCS_FILENAME, _META_FILENAME):
            if not (path / required).is_file():
                raise FileNotFoundError(f"Missing {required} in {path}")

        with open(path / _META_FILENAME, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(path / _DOCS_FILENAME, "rb") as fh:
            documents = pickle.load(fh)
        index = faiss.read_index(str(path / _INDEX_FILENAME))
        # Reconstructing the embedding matrix keeps subsequent ``add_documents``
        # calls consistent (the FAISS index alone doesn't expose vectors back).
        embeddings = index.reconstruct_n(0, index.ntotal).astype(np.float32)

        retriever = cls(
            model_name=meta["model_name"],
            batch_size=meta.get("batch_size", 32),
        )
        retriever._restore_from(documents, index, embeddings)
        logger.info("Loaded DenseRetriever (%d docs) from %s", len(documents), path)
        return retriever

    def __len__(self) -> int:
        return len(self._documents)
