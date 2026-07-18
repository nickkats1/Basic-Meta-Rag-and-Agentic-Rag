"""Load and chunk documents for RAG pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from pypdf import PdfReader


@dataclass
class Document:
    """A chunk of text with metadata."""

    page_content: str
    metadata: dict


def load_documents(file_path: Union[str, Path]) -> list[Document]:
    """Load text from a PDF file."""
    path = Path(file_path)
    reader = PdfReader(str(path))
    docs = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": page_num},
            )
        )
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
) -> list[Document]:
    """Split documents into fixed-size chunks with overlap."""
    chunks: list[Document] = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk_text = text[i : i + chunk_size]
            if chunk_text.strip():
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            **doc.metadata,
                            "doc_id": f"chunk_{len(chunks)}",
                        },
                    )
                )
    return chunks
