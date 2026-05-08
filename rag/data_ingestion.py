"""PDF loading and chunking utilities.

Two narrow responsibilities:

* :func:`load_documents` -- read a PDF from disk into a list of LangChain
  ``Document`` objects (one per page).
* :func:`chunk_documents` -- split documents with a recursive character
  splitter so they fit inside an embedding context window.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


PathLike = Union[str, Path]


def load_documents(file_path: PathLike) -> List[Document]:
    """Load a PDF file into a list of ``Document`` objects, one per page.

    Args:
        file_path: Filesystem path to a PDF.

    Returns:
        One :class:`langchain_core.documents.Document` per PDF page, with
        page metadata populated by ``PyPDFLoader``.

    Raises:
        FileNotFoundError: If ``file_path`` is ``None`` or the file does not
            exist on disk.
    """
    if file_path is None:
        raise FileNotFoundError("file_path must not be None")

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found at: {path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()
    logger.info("Loaded %d page(s) from %s", len(documents), path)
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    assign_doc_ids: bool = True,
    doc_id_prefix: str = "chunk",
) -> List[Document]:
    """Split documents into overlapping chunks.

    Args:
        documents: Documents to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlapping characters between adjacent chunks.
        assign_doc_ids: If ``True`` (the default), each chunk gets a stable
            ``metadata['doc_id']`` of the form ``{doc_id_prefix}_{index}``.
            Set to ``False`` to leave existing metadata untouched.
        doc_id_prefix: Prefix used when ``assign_doc_ids`` is ``True``.

    Returns:
        Chunked :class:`Document` objects with their original metadata
        propagated by ``RecursiveCharacterTextSplitter``.

    Raises:
        ValueError: If ``documents`` is empty, ``chunk_size`` is non-positive,
            ``chunk_overlap`` is negative, or ``chunk_overlap >= chunk_size``.
    """
    if not documents:
        raise ValueError("No documents provided")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "\t", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if assign_doc_ids:
        for i, chunk in enumerate(chunks):
            chunk.metadata["doc_id"] = f"{doc_id_prefix}_{i}"

    logger.info("Split %d document(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks
