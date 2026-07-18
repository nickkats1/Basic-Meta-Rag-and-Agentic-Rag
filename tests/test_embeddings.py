"""Tests for embedding functions."""

from __future__ import annotations

import numpy as np

from rag.embeddings import embed_texts, load_bi_encoder, load_cross_encoder


class TestLoadBiEncoder:
    """Test bi-encoder loading."""

    def test_load_bi_encoder_returns_model(self):
        """load_bi_encoder returns a SentenceTransformer model."""
        model = load_bi_encoder()
        assert model is not None

    def test_load_bi_encoder_custom_model(self):
        """load_bi_encoder accepts custom model name."""
        model = load_bi_encoder("sentence-transformers/all-MiniLM-L6-v2")
        assert model is not None


class TestLoadCrossEncoder:
    """Test cross-encoder loading."""

    def test_load_cross_encoder_returns_model(self):
        """load_cross_encoder returns a CrossEncoder model."""
        model = load_cross_encoder()
        assert model is not None

    def test_load_cross_encoder_custom_model(self):
        """load_cross_encoder accepts custom model name."""
        model = load_cross_encoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert model is not None


class TestEmbedTexts:
    """Test text embedding."""

    def test_embed_texts_returns_array(self):
        """embed_texts returns numpy array."""
        texts = ["Hello world", "How are you"]
        embeddings = embed_texts(texts)
        assert isinstance(embeddings, np.ndarray)

    def test_embed_texts_correct_shape(self):
        """embed_texts returns (n, dim) array."""
        texts = ["Text one", "Text two", "Text three"]
        embeddings = embed_texts(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] > 0

    def test_embed_texts_float32(self):
        """embed_texts returns float32 array."""
        texts = ["Sample text"]
        embeddings = embed_texts(texts)
        assert embeddings.dtype == np.float32

    def test_embed_texts_normalized(self):
        """embed_texts returns normalized embeddings."""
        texts = ["First text", "Second text"]
        embeddings = embed_texts(texts)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(texts)))

    def test_embed_texts_single_text(self):
        """embed_texts works with single text."""
        texts = ["Single text"]
        embeddings = embed_texts(texts)
        assert embeddings.shape == (1, embeddings.shape[1])

    def test_embed_texts_custom_batch_size(self):
        """embed_texts accepts custom batch size."""
        texts = ["Text " + str(i) for i in range(10)]
        embeddings = embed_texts(texts, batch_size=2)
        assert embeddings.shape[0] == 10
