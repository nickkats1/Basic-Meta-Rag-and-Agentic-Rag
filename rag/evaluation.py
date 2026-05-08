"""Evaluation harness for comparing retrievers and full RAG pipelines.

Two entry points:

* :func:`evaluate_retriever` -- runs a single retriever over a labeled
  dataset and returns retrieval metrics.
* :func:`evaluate_pipeline` -- runs an end-to-end pipeline and returns both
  retrieval and (optionally) generation metrics.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from rag.metrics.generation import (
    answer_similarity,
    context_precision,
    embedding_faithfulness,
)
from rag.metrics.retrieval import (
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.pipeline import RAGPipeline
from rag.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """A single labeled evaluation example.

    Attributes:
        question: The user query.
        relevant_doc_ids: Ground-truth relevant document ids.
        reference_answer: Optional gold answer for generation metrics.
    """

    question: str
    relevant_doc_ids: List[str]
    reference_answer: Optional[str] = None


@dataclass
class EvalReport:
    """Aggregated evaluation results.

    Attributes:
        metrics: Mean of each metric across the dataset.
        per_example: Per-example metric values, in the same order as the input
            dataset.
    """

    metrics: Dict[str, float]
    per_example: List[Dict[str, float]] = field(default_factory=list)


DocIdFn = Callable[[object], str]


def _default_doc_id(document) -> str:
    """Default document-id extractor.

    Uses ``metadata['doc_id']`` when present, falling back to
    ``page_content``. Exported as :func:`default_doc_id` for callers that want
    to compose with custom extractors.
    """
    if document.metadata and "doc_id" in document.metadata:
        return str(document.metadata["doc_id"])
    return document.page_content


# Public alias.
default_doc_id = _default_doc_id


def evaluate_retriever(
    retriever: BaseRetriever,
    dataset: Sequence[EvalExample],
    k: int = 5,
    doc_id_fn: DocIdFn = _default_doc_id,
) -> EvalReport:
    """Evaluate a retriever's ranking quality on a labeled dataset.

    Args:
        retriever: Retriever under test. Must already be indexed.
        dataset: Labeled examples.
        k: Cutoff for ``@k`` metrics.
        doc_id_fn: Maps a retrieved ``Document`` to its identifier so that it
            can be compared with ``relevant_doc_ids``.

    Returns:
        :class:`EvalReport` with hit rate, recall, precision, MRR, NDCG, and
        MAP averaged across the dataset.
    """
    per_example: List[Dict[str, float]] = []
    batch_retrieved: List[List[str]] = []
    batch_relevant: List[List[str]] = []

    for example in dataset:
        hits = retriever.retrieve_documents(example.question, k=k)
        retrieved_ids = [doc_id_fn(doc) for doc in hits]
        batch_retrieved.append(retrieved_ids)
        batch_relevant.append(example.relevant_doc_ids)

        per_example.append(
            {
                f"hit@{k}": hit_rate_at_k(retrieved_ids, example.relevant_doc_ids, k),
                f"precision@{k}": precision_at_k(retrieved_ids, example.relevant_doc_ids, k),
                f"recall@{k}": recall_at_k(retrieved_ids, example.relevant_doc_ids, k),
                f"ndcg@{k}": ndcg_at_k(retrieved_ids, example.relevant_doc_ids, k),
                "mrr": mean_reciprocal_rank(retrieved_ids, example.relevant_doc_ids),
            }
        )

    aggregate = {
        key: sum(item[key] for item in per_example) / max(len(per_example), 1)
        for key in (per_example[0].keys() if per_example else [])
    }
    aggregate["map"] = mean_average_precision(batch_retrieved, batch_relevant)
    return EvalReport(metrics=aggregate, per_example=per_example)


def evaluate_pipeline(
    pipeline: RAGPipeline,
    dataset: Sequence[EvalExample],
    k: int = 5,
    doc_id_fn: DocIdFn = _default_doc_id,
    include_generation: bool = True,
    faithfulness_threshold: float = 0.5,
) -> EvalReport:
    """Evaluate a full RAG pipeline (retrieval + generation).

    Args:
        pipeline: The pipeline under test.
        dataset: Labeled examples. ``reference_answer`` is required for
            generation metrics.
        k: Cutoff for ``@k`` retrieval metrics.
        doc_id_fn: Maps retrieved ``Document`` objects to ids.
        include_generation: If ``True``, compute answer similarity,
            embedding faithfulness, and context precision when reference
            answers are available.
        faithfulness_threshold: Threshold passed to
            :func:`context_precision`.

    Returns:
        :class:`EvalReport` covering retrieval and (optionally) generation
        metrics.
    """
    per_example: List[Dict[str, float]] = []
    batch_retrieved: List[List[str]] = []
    batch_relevant: List[List[str]] = []

    for example in dataset:
        response = pipeline.answer(example.question)
        retrieved_ids = [doc_id_fn(doc) for doc in response.contexts]
        batch_retrieved.append(retrieved_ids)
        batch_relevant.append(example.relevant_doc_ids)

        record: Dict[str, float] = {
            f"hit@{k}": hit_rate_at_k(retrieved_ids, example.relevant_doc_ids, k),
            f"precision@{k}": precision_at_k(retrieved_ids, example.relevant_doc_ids, k),
            f"recall@{k}": recall_at_k(retrieved_ids, example.relevant_doc_ids, k),
            f"ndcg@{k}": ndcg_at_k(retrieved_ids, example.relevant_doc_ids, k),
            "mrr": mean_reciprocal_rank(retrieved_ids, example.relevant_doc_ids),
        }

        if include_generation and example.reference_answer:
            context_strings = [doc.page_content for doc in response.contexts]
            record["answer_similarity"] = answer_similarity(
                response.answer, example.reference_answer
            )
            record["faithfulness"] = embedding_faithfulness(
                response.answer, context_strings
            )
            record["context_precision"] = context_precision(
                example.reference_answer,
                context_strings,
                threshold=faithfulness_threshold,
            )

        per_example.append(record)

    aggregate = {
        key: sum(item.get(key, 0.0) for item in per_example) / max(len(per_example), 1)
        for key in (per_example[0].keys() if per_example else [])
    }
    aggregate["map"] = mean_average_precision(batch_retrieved, batch_relevant)
    return EvalReport(metrics=aggregate, per_example=per_example)
