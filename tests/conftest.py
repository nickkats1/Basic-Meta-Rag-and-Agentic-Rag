"""Shared pytest fixtures for the Sec-Rag test suite."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag.data_ingestion import Document


@pytest.fixture()
def pdf_file(tmp_path: Path) -> Path:
    """A single minimal one-page PDF file."""
    pdf_path = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


@pytest.fixture()
def pdf_directory(tmp_path: Path) -> Path:
    """A directory containing a single minimal PDF file."""
    pdf_dir = tmp_path / "data"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_dir


@pytest.fixture()
def sample_documents() -> list[Document]:
    """A small list of Document objects for unit testing."""
    return [
        Document(
            page_content="Hello. " * 50,
            metadata={"source": "test.pdf", "page": 0},
        ),
        Document(
            page_content="more content. " * 50,
            metadata={"source": "test.pdf", "page": 1},
        ),
    ]