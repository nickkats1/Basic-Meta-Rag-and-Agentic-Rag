"""Input-validation tests for :func:`rag.embeddings.embed_texts`.

These tests deliberately avoid the embedding path so no model is downloaded.
"""
from __future__ import annotations

import pytest

from rag.embeddings import embed_texts


class TestEmbedTextsInputValidation:
    """Argument validation for ``embed_texts`` happens before model load."""

    def test_single_string_raises_type_error(self):
        with pytest.raises(TypeError):
            embed_texts("hello world")

    def test_empty_sequence_raises_value_error(self):
        with pytest.raises(ValueError):
            embed_texts([])

    def test_empty_tuple_raises_value_error(self):
        with pytest.raises(ValueError):
            embed_texts(())
