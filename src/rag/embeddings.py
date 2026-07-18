"""Embedding models using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


@lru_cache(maxsize=None)
def load_bi_encoder(model_name: str = DEFAULT_BI_ENCODER):
    """Load a bi-encoder model, cached. Falls back to CPU if the GPU is full."""
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name)
    except RuntimeError as error:
        if not _is_oom(error):
            raise
        return SentenceTransformer(model_name, device="cpu")




def load_cross_encoder(model_name: str = DEFAULT_CROSS_ENCODER):
    """Load a cross-encoder model, cached. Falls back to CPU if the GPU is full."""
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(model_name)
    except RuntimeError as error:
        if not _is_oom(error):
            raise
        return CrossEncoder(model_name, device="cpu")


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_BI_ENCODER,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed texts to (n, dim) float32 array."""
    model = load_bi_encoder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)


def score_pairs(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    model_name: str = DEFAULT_CROSS_ENCODER,
) -> np.ndarray:
    """Score query-document pairs. Returns (n,) array."""
    model = load_cross_encoder(model_name)
    pairs = [(query_embedding, doc_emb) for doc_emb in doc_embeddings]
    scores = model.predict(pairs, convert_to_numpy=True)
    return scores
