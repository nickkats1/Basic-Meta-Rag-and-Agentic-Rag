"""Tests for :mod:`rag.data_ingestion`."""

from __future__ import annotations


from rag.data_ingestion import Document, chunk_documents, load_documents


class TestLoadDocuments:
    """load_documents reads PDFs."""

    def test_loads_blank_pdf(self, pdf_file):
        docs = load_documents(str(pdf_file))
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_accepts_pathlib(self, pdf_file):
        docs = load_documents(pdf_file)
        assert len(docs) > 0


class TestChunkDocuments:
    """chunk_documents splits documents."""

    def test_splits_into_chunks(self):
        long_text = "a b c d " * 200
        docs = [Document(page_content=long_text, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        docs = [Document(page_content="hello world", metadata={"source": "test.pdf"})]
        chunks = chunk_documents(docs, chunk_size=5, chunk_overlap=0)
        assert all("source" in c.metadata for c in chunks)

