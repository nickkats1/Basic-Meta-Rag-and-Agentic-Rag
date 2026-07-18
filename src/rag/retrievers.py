"""Retrieval strategies for RAG."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from rag.data_ingestion import Document
from rag.embeddings import embed_texts, load_cross_encoder


class Retriever(ABC):
    """Base retriever interface."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None:
        """Index documents."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top-k documents for a query."""


class BM25Retriever(Retriever):
    """Sparse retrieval using BM25."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None

    def add_documents(self, documents: list[Document]) -> None:
        self.documents = documents
        tokenized = [doc.page_content.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if self.bm25 is None:
            return []
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in indices]


class DenseRetriever(Retriever):
    """Dense retrieval using bi-encoders + FAISS."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None
        self.index: Any = None

    def add_documents(self, documents: list[Document]) -> None:
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = embed_texts(texts, model_name=self.model_name)
        self._build_index()

    def _build_index(self) -> None:
        if self.embeddings is None:
            return
        try:
            import faiss

            dim = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        except ImportError:
            pass

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if self.embeddings is None:
            return []
        query_emb = embed_texts([query], model_name=self.model_name)
        if self.index is not None:
            _, indices = self.index.search(query_emb, top_k)
            return [self.documents[i] for i in indices[0] if i < len(self.documents)]
        distances = np.dot(query_emb, self.embeddings.T)[0]
        indices = np.argsort(distances)[::-1][:top_k]
        return [self.documents[i] for i in indices]

    def save(self, path: str | Path) -> None:
        """Persist embeddings and index."""
        import faiss
        import pickle

        if self.embeddings is None:
            raise ValueError("Nothing to save: call add_documents() first.")
        path = Path(path)
        path.mkdir(exist_ok=True)
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        np.save(path / "embeddings.npy", self.embeddings)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))

    def load(self, path: str | Path) -> None:
        """Load persisted embeddings and index."""
        import faiss
        import pickle

        path = Path(path)
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        self.embeddings = np.load(path / "embeddings.npy")
        if (path / "index.faiss").exists():
            self.index = faiss.read_index(str(path / "index.faiss"))


class HybridRetriever(Retriever):
    """Combines BM25 + dense with Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_weight: float = 1.0,
        dense_weight: float = 1.0,
        candidates_per_retriever: int = 40,
    ):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidates_per_retriever = candidates_per_retriever
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()

    def add_documents(self, documents: list[Document]) -> None:
        self.bm25.add_documents(documents)
        self.dense.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        bm25_results = self.bm25.retrieve(query, top_k=self.candidates_per_retriever)
        dense_results = self.dense.retrieve(query, top_k=self.candidates_per_retriever)

        bm25_map = {doc.metadata["doc_id"]: i for i, doc in enumerate(bm25_results)}
        dense_map = {doc.metadata["doc_id"]: i for i, doc in enumerate(dense_results)}

        all_ids = set(bm25_map.keys()) | set(dense_map.keys())
        scores = {}
        for doc_id in all_ids:
            bm25_rank = bm25_map.get(doc_id, self.candidates_per_retriever)
            dense_rank = dense_map.get(doc_id, self.candidates_per_retriever)
            score = (
                self.bm25_weight / (bm25_rank + 60)
                + self.dense_weight / (dense_rank + 60)
            )
            scores[doc_id] = score

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[
            :top_k
        ]
        result_map = {
            doc.metadata["doc_id"]: doc
            for doc in bm25_results + dense_results
            if doc.metadata["doc_id"] in sorted_ids
        }
        return [result_map[doc_id] for doc_id in sorted_ids if doc_id in result_map]


class RerankerRetriever(Retriever):
    """Wraps a base retriever with cross-encoder reranking."""

    def __init__(
        self,
        base_retriever: Retriever,
        candidate_pool: int = 20,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.base = base_retriever
        self.candidate_pool = candidate_pool
        self.model_name = model_name

    def add_documents(self, documents: list[Document]) -> None:
        self.base.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        candidates = self.base.retrieve(query, top_k=self.candidate_pool)
        if not candidates:
            return []

        model = load_cross_encoder(self.model_name)
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = model.predict(pairs, convert_to_numpy=True)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
