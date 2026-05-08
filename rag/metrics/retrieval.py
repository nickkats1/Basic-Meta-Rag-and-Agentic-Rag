"""Standard order-aware retrieval metrics.

All functions accept either:

* ``retrieved`` as a list of identifier strings (e.g. document ids) and
  ``relevant`` as a set/iterable of identifier strings, or
* dataset-level batches via the ``mean_*`` aggregators.

Identifiers are compared with ``==``; pick something stable (the chunk's hash,
filename + page number, etc.) before passing them in.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _truncate(retrieved: Sequence[str], k: int) -> Sequence[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return retrieved[:k]


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return ``1.0`` if any relevant document appears in the top ``k``, else ``0.0``.

    Args:
        retrieved: Ranked list of retrieved document ids.
        relevant: Ground-truth relevant document ids.
        k: Cutoff.
    """
    relevant_set = set(relevant)
    return float(any(doc_id in relevant_set for doc_id in _truncate(retrieved, k)))


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top ``k`` retrieved documents that are relevant.

    Args:
        retrieved: Ranked list of retrieved document ids.
        relevant: Ground-truth relevant document ids.
        k: Cutoff.
    """
    relevant_set = set(relevant)
    top_k = _truncate(retrieved, k)
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents that appear in the top ``k``.

    Args:
        retrieved: Ranked list of retrieved document ids.
        relevant: Ground-truth relevant document ids.
        k: Cutoff.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top_k = _truncate(retrieved, k)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def mean_reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Reciprocal rank of the first relevant hit, or ``0`` if none.

    Args:
        retrieved: Ranked list of retrieved document ids.
        relevant: Ground-truth relevant document ids.
    """
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Average Precision (AP), i.e. mean of P@k at each relevant rank.

    Args:
        retrieved: Ranked list of retrieved document ids.
        relevant: Ground-truth relevant document ids.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = 0
    cumulative = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            hits += 1
            cumulative += hits / rank
    return cumulative / len(relevant_set)


def mean_average_precision(
    batch_retrieved: Sequence[Sequence[str]],
    batch_relevant: Sequence[Iterable[str]],
) -> float:
    """Mean Average Precision (MAP) across a batch of queries.

    Args:
        batch_retrieved: One ranked list per query.
        batch_relevant: One iterable of relevant ids per query, aligned with
            ``batch_retrieved``.

    Raises:
        ValueError: If the two arguments have different lengths.
    """
    if len(batch_retrieved) != len(batch_relevant):
        raise ValueError("batch_retrieved and batch_relevant must align in length")
    if not batch_retrieved:
        return 0.0
    return sum(
        average_precision(r, g) for r, g in zip(batch_retrieved, batch_relevant)
    ) / len(batch_retrieved)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at cutoff ``k`` (binary relevance).

    Uses the standard NDCG formulation:
    ``DCG = sum_i rel_i / log2(i + 1)`` with ``i`` 1-indexed, divided by the
    ideal DCG given the available number of relevant documents.

    Args:
        retrieved: Ranked list of retrieved document ids.
        relevant: Ground-truth relevant document ids (binary).
        k: Cutoff.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    top_k = _truncate(retrieved, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(top_k, start=1)
        if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
