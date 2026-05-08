"""Embedding model wrappers built directly on ``sentence-transformers``.

This module avoids the LangChain abstraction layer so that callers can work
with raw ``numpy`` arrays and have direct control over batching, device
placement, and normalization. The ``DenseRetriever`` and the cross-encoder
reranker both depend on the model loaders defined here.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=4)
def load_bi_encoder(model_name: str = DEFAULT_BI_ENCODER):
    """Load a sentence-transformers bi-encoder, cached per model name.

    Args:
        model_name: HuggingFace Hub model id of a bi-encoder.

    Returns:
        A ``sentence_transformers.SentenceTransformer`` instance.
    """
    from sentence_transformers import SentenceTransformer

    logger.info("Loading bi-encoder: %s", model_name)
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER):
    """Load a sentence-transformers cross-encoder, cached per model name.

    Args:
        model_name: HuggingFace Hub model id of a cross-encoder.

    Returns:
        A ``sentence_transformers.CrossEncoder`` instance.
    """
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder: %s", model_name)
    return CrossEncoder(model_name)


def embed_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_BI_ENCODER,
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Embed a batch of texts with a bi-encoder.

    Args:
        texts: Strings to embed.
        model_name: HuggingFace Hub model id.
        batch_size: Encoder batch size.
        normalize: If ``True``, L2-normalize embeddings so that dot-product
            equals cosine similarity.

    Returns:
        A ``(len(texts), dim)`` ``numpy.ndarray`` of float32 embeddings.

    Raises:
        ValueError: If ``texts`` is empty.
    """
    if isinstance(texts, str):
        raise TypeError(
            "texts must be a sequence of strings, not a single str "
            "(passing one string would embed each character)"
        )
    if not texts:
        raise ValueError("texts must be a non-empty sequence")

    model = load_bi_encoder(model_name)
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32, copy=False)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two batches of vectors.

    Args:
        a: ``(n, d)`` array.
        b: ``(m, d)`` array.

    Returns:
        ``(n, m)`` matrix of cosine similarities.
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b_norm = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return (a / a_norm) @ (b / b_norm).T


__all__ = [
    "DEFAULT_BI_ENCODER",
    "DEFAULT_CROSS_ENCODER",
    "cosine_similarity",
    "embed_texts",
    "load_bi_encoder",
    "load_cross_encoder",
]
