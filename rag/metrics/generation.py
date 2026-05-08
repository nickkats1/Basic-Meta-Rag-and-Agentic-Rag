"""Generation-quality metrics that don't require an LLM judge.

Three lightweight, embedding-based metrics:

* :func:`answer_similarity` -- cosine similarity between predicted and
  reference answers.
* :func:`embedding_faithfulness` -- the maximum cosine of the predicted
  answer to any retrieved context chunk; high values indicate the answer is
  semantically grounded in the context.
* :func:`context_precision` -- fraction of retrieved contexts that are
  semantically close (>= ``threshold``) to the reference answer; a proxy for
  whether the retriever is surfacing the right passages.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rag.embeddings import DEFAULT_BI_ENCODER, cosine_similarity, embed_texts


def answer_similarity(
    predicted: str,
    reference: str,
    model_name: str = DEFAULT_BI_ENCODER,
) -> float:
    """Cosine similarity between embeddings of two answers.

    Args:
        predicted: The model-generated answer.
        reference: The gold answer.
        model_name: Bi-encoder model id used to embed both strings.

    Returns:
        Cosine similarity in ``[-1, 1]``. Higher is more similar.
    """
    if not predicted.strip() or not reference.strip():
        return 0.0
    embeddings = embed_texts([predicted, reference], model_name=model_name)
    return float(cosine_similarity(embeddings[:1], embeddings[1:])[0, 0])


def embedding_faithfulness(
    predicted: str,
    contexts: Sequence[str],
    model_name: str = DEFAULT_BI_ENCODER,
) -> float:
    """Maximum cosine similarity between the answer and any context chunk.

    A faithful answer should be close to at least one passage that was given
    to the model. This is a cheap, LLM-free proxy for the
    "is the answer grounded?" question.

    Args:
        predicted: The model-generated answer.
        contexts: Retrieved context strings shown to the model.
        model_name: Bi-encoder used for embedding.

    Returns:
        ``max_i cosine(answer, context_i)``, or ``0.0`` if either input is empty.
    """
    if not predicted.strip() or not contexts:
        return 0.0
    answer_emb = embed_texts([predicted], model_name=model_name)
    context_emb = embed_texts(list(contexts), model_name=model_name)
    return float(np.max(cosine_similarity(answer_emb, context_emb)))


def context_precision(
    reference: str,
    contexts: Sequence[str],
    threshold: float = 0.5,
    model_name: str = DEFAULT_BI_ENCODER,
) -> float:
    """Fraction of contexts whose cosine to the reference answer exceeds ``threshold``.

    Args:
        reference: The gold answer or evidence string.
        contexts: Retrieved context strings.
        threshold: Cosine threshold above which a context counts as relevant.
        model_name: Bi-encoder used for embedding.

    Returns:
        ``relevant_contexts / total_contexts`` in ``[0, 1]``.
    """
    if not reference.strip() or not contexts:
        return 0.0
    ref_emb = embed_texts([reference], model_name=model_name)
    context_emb = embed_texts(list(contexts), model_name=model_name)
    sims = cosine_similarity(ref_emb, context_emb)[0]
    return float(np.mean(sims >= threshold))
