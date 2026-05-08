"""Tests for :mod:`rag.evaluation` helpers."""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.evaluation import default_doc_id


class TestDefaultDocId:
    """Behavior of the public ``default_doc_id`` extractor."""

    def test_uses_metadata_doc_id_when_present(self):
        doc = Document(page_content="content", metadata={"doc_id": "abc"})
        assert default_doc_id(doc) == "abc"

    def test_falls_back_to_page_content_when_metadata_missing_doc_id(self):
        doc = Document(page_content="some content", metadata={"source": "f.pdf"})
        assert default_doc_id(doc) == "some content"

    def test_falls_back_to_page_content_when_metadata_empty(self):
        doc = Document(page_content="payload", metadata={})
        assert default_doc_id(doc) == "payload"

    def test_falls_back_to_page_content_when_metadata_none(self):
        doc = Document(page_content="payload")
        # langchain's Document defaults metadata to {}, but force-clear to
        # cover the explicit-None branch.
        doc.metadata = None
        assert default_doc_id(doc) == "payload"

    def test_returns_str_even_when_doc_id_is_int(self):
        doc = Document(page_content="content", metadata={"doc_id": 42})
        result = default_doc_id(doc)
        assert result == "42"
        assert isinstance(result, str)

    @pytest.mark.parametrize("doc_id", [1, 2.5, True])
    def test_non_string_doc_ids_are_coerced(self, doc_id):
        doc = Document(page_content="x", metadata={"doc_id": doc_id})
        assert isinstance(default_doc_id(doc), str)
