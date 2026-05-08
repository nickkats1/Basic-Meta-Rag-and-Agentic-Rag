"""Shared pytest fixtures."""
from __future__ import annotations

from typing import List

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter


@pytest.fixture()
def pdf_file(tmp_path):
    """Yield a path to a minimal one-page PDF on disk."""
    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


@pytest.fixture()
def toy_corpus() -> List[Document]:
    """A tiny labeled corpus used across retriever tests.

    Each document carries a ``doc_id`` in its metadata so retrieval metrics
    have a stable identifier to score against.
    """
    raw = [
        ("a", "Apple reported record iPhone revenue in Q4 2024."),
        ("b", "Tesla delivered 1.8 million electric vehicles last year."),
        ("c", "The Federal Reserve raised interest rates by 25 basis points."),
        ("d", "Microsoft's cloud business grew 30% year over year."),
        ("e", "Amazon Web Services launched a new AI inference service."),
        ("f", "Google's advertising revenue declined slightly in Q1."),
    ]
    return [
        Document(page_content=text, metadata={"doc_id": doc_id})
        for doc_id, text in raw
    ]
