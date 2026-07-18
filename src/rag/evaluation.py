"""Evaluation harness for RAG systems."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.metrics import (
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.pipeline import RAGPipeline
from rag.retrievers import Retriever


@dataclass
class EvalExample:
    """Evaluation example with question and gold answers."""

    question: str
    relevant_doc_ids: list[str]
    reference_answer: str = ""


@dataclass
class EvalResult:
    """Evaluation results."""

    metrics: dict[str, float] = field(default_factory=dict)
    per_example: list[dict[str, float]] = field(default_factory=list)


def evaluate_retriever(
    retriever: Retriever,
    examples: list[EvalExample],
    k: int = 5,
) -> EvalResult:
    """Evaluate retriever on examples."""
    all_retrieved = []
    all_relevant = []
    per_example = []

    for example in examples:
        docs = retriever.retrieve(example.question, top_k=k)
        retrieved_ids = [doc.metadata.get("doc_id", "") for doc in docs]
        all_retrieved.append(retrieved_ids)
        all_relevant.append(example.relevant_doc_ids)

        per_ex = {
            "hit": hit_rate_at_k(retrieved_ids, example.relevant_doc_ids, k),
            "precision": precision_at_k(retrieved_ids, example.relevant_doc_ids, k),
            "recall": recall_at_k(retrieved_ids, example.relevant_doc_ids, k),
            "mrr": mean_reciprocal_rank(retrieved_ids, example.relevant_doc_ids),
            "ndcg": ndcg_at_k(retrieved_ids, example.relevant_doc_ids, k),
            "ap": average_precision(retrieved_ids, example.relevant_doc_ids),
        }
        per_example.append(per_ex)

    metrics = {
        "hit@" + str(k): sum(ex["hit"] for ex in per_example) / len(per_example),
        "precision@" + str(k): sum(ex["precision"] for ex in per_example)
        / len(per_example),
        "recall@" + str(k): sum(ex["recall"] for ex in per_example) / len(per_example),
        "mrr": sum(ex["mrr"] for ex in per_example) / len(per_example),
        "ndcg@" + str(k): sum(ex["ndcg"] for ex in per_example) / len(per_example),
        "map": mean_average_precision(all_retrieved, all_relevant),
    }
    return EvalResult(metrics=metrics, per_example=per_example)


def evaluate_pipeline(
    pipeline: RAGPipeline,
    examples: list[EvalExample],
    k: int = 5,
) -> EvalResult:
    """Evaluate RAG pipeline on examples."""
    result = evaluate_retriever(pipeline.retriever, examples, k=k)

    if not any(ex.reference_answer for ex in examples):
        return result

    for i, example in enumerate(examples):
        if not example.reference_answer:
            continue
        rag_result = pipeline.answer(example.question)
        from rag.metrics import answer_similarity

        result.per_example[i]["answer_sim"] = answer_similarity(
            example.reference_answer, rag_result.answer
        )

    if any("answer_sim" in ex for ex in result.per_example):
        result.metrics["answer_similarity"] = sum(
            ex.get("answer_sim", 0) for ex in result.per_example
        ) / len(result.per_example)

    return result
