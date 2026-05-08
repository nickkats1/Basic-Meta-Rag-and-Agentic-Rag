"""Tests for :mod:`rag.data_ingestion`."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.data_ingestion import chunk_documents, load_documents


class TestLoadDocuments:
    """``load_documents`` reads PDFs and rejects bad inputs."""

    def test_loads_blank_pdf(self, pdf_file):
        docs = load_documents(str(pdf_file))
        assert len(docs) > 0

    def test_accepts_pathlib(self, pdf_file):
        docs = load_documents(pdf_file)  # ``Path`` directly
        assert len(docs) > 0

    def test_raises_on_none(self):
        with pytest.raises(FileNotFoundError):
            load_documents(file_path=None)

    def test_raises_on_missing_path(self, tmp_path):
        missing = tmp_path / "does-not-exist.pdf"
        with pytest.raises(FileNotFoundError):
            load_documents(str(missing))


class TestChunkDocuments:
    """``chunk_documents`` validates inputs and produces chunks."""

    def test_splits_into_smaller_chunks(self):
        long_text = "a b c d " * 200
        docs = [Document(page_content=long_text)]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 50 for c in chunks)

    def test_rejects_empty_documents(self):
        with pytest.raises(ValueError):
            chunk_documents([], chunk_size=100, chunk_overlap=10)

    def test_rejects_overlap_at_or_above_chunk_size(self):
        docs = [Document(page_content="hello world")]
        with pytest.raises(ValueError):
            chunk_documents(docs, chunk_size=10, chunk_overlap=10)

    def test_rejects_non_positive_chunk_size(self):
        docs = [Document(page_content="hello")]
        with pytest.raises(ValueError):
            chunk_documents(docs, chunk_size=0, chunk_overlap=0)

    def test_rejects_negative_overlap(self):
        docs = [Document(page_content="hello")]
        with pytest.raises(ValueError):
            chunk_documents(docs, chunk_size=10, chunk_overlap=-1)

    def test_assigns_doc_ids_by_default(self):
        long_text = "a b c d " * 200
        docs = [Document(page_content=long_text)]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        assert all("doc_id" in c.metadata for c in chunks)
        assert chunks[0].metadata["doc_id"] == "chunk_0"
        assert chunks[-1].metadata["doc_id"] == f"chunk_{len(chunks) - 1}"

    def test_doc_ids_can_be_disabled(self):
        long_text = "a b c d " * 200
        docs = [Document(page_content=long_text)]
        chunks = chunk_documents(
            docs, chunk_size=50, chunk_overlap=10, assign_doc_ids=False
        )
        assert all("doc_id" not in c.metadata for c in chunks)

    def test_doc_id_prefix_respected(self):
        long_text = "a b c d " * 200
        docs = [Document(page_content=long_text)]
        chunks = chunk_documents(
            docs, chunk_size=50, chunk_overlap=10, doc_id_prefix="g10k"
        )
        assert chunks[0].metadata["doc_id"] == "g10k_0"
