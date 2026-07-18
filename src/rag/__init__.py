"""RAG package for SEC filings."""

from rag.data_ingestion import Document, chunk_documents, load_documents
from rag.evaluation import EvalExample, EvalResult, evaluate_pipeline, evaluate_retriever
from rag.llm import GroqLLM, GoogleLLM, HuggingFaceLocalLLM, OpenAILLM, get_llm
from rag.metrics import (
    answer_similarity,
    average_precision,
    hit_rate_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag.pipeline import RAGPipeline, RAGResult
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)

__all__ = [
    "Document",
    "load_documents",
    "chunk_documents",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankerRetriever",
    "OpenAILLM",
    "GroqLLM",
    "GoogleLLM",
    "HuggingFaceLocalLLM",
    "get_llm",
    "RAGPipeline",
    "RAGResult",
    "EvalExample",
    "EvalResult",
    "evaluate_retriever",
    "evaluate_pipeline",
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "average_precision",
    "mean_average_precision",
    "ndcg_at_k",
    "answer_similarity",
]
