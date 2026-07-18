"""Retrieval and generation metrics."""

from __future__ import annotations

import numpy as np


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of queries with at least one relevant doc in top-k."""
    if not relevant_ids:
        return 0.0
    return float(any(doc_id in retrieved_ids[:k] for doc_id in relevant_ids))


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of top-k that are relevant."""
    if k == 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return sum(1 for doc_id in top_k if doc_id in relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return sum(1 for doc_id in relevant_ids if doc_id in top_k) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal of rank of first relevant doc."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Average precision across all relevant documents."""
    if not relevant_ids:
        return 0.0
    precisions = []
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            precisions.append(precision_at_k(retrieved_ids, relevant_ids, k=i + 1))
    return float(np.mean(precisions)) if precisions else 0.0


def mean_average_precision(
    all_retrieved: list[list[str]], all_relevant: list[list[str]]
) -> float:
    """Average precision across all queries."""
    if not all_retrieved:
        return 0.0
    aps = [
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    ]
    return float(np.mean(aps))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


def answer_similarity(reference: str, generated: str) -> float:
    """Cosine similarity between reference and generated answers."""
    from rag.embeddings import embed_texts

    embeddings = embed_texts([reference, generated])
    return float(np.dot(embeddings[0], embeddings[1]))
