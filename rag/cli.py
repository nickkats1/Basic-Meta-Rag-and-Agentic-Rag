"""Command-line entry point: ``python -m rag.cli ...``.

Two subcommands:

* ``answer`` -- run a single query through one retriever + one LLM
* ``compare`` -- run the same query through all four retrievers and print a
  retrieval-metrics table (uses the cross-encoder reranker as a stand-in
  oracle when no ``--gold-id`` is provided).

Examples::

    python -m rag.cli answer \\
        --pdf data/google_10K.pdf --retriever reranker \\
        --query "What was total revenue in each fiscal year reported?"

    python -m rag.cli compare \\
        --pdf data/google_10K.pdf \\
        --query "Operating income for each reported year"
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List

from dotenv import load_dotenv

from rag.data_ingestion import chunk_documents, load_documents
from rag.evaluation import EvalExample, evaluate_retriever
from rag.llm import LLM, SUPPORTED_PROVIDERS
from rag.metrics import embedding_faithfulness
from rag.pipeline import RAGPipeline
from rag.retrievers import (
    BM25Retriever,
    BaseRetriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)


RETRIEVERS = ("bm25", "dense", "hybrid", "reranker")


def _build_retriever(name: str) -> BaseRetriever:
    """Factory for the four retrievers, keyed by short name."""
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return DenseRetriever()
    if name == "hybrid":
        return HybridRetriever()
    if name == "reranker":
        return RerankerRetriever(base_retriever=HybridRetriever(), candidate_pool=20)
    raise ValueError(f"Unknown retriever: {name!r}. Choose one of {RETRIEVERS}.")


_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "google": "GOOGLE_API_KEY",
    "huggingface": "HF_TOKEN",
    "huggingface_local": "HF_TOKEN",
}


def _provider_api_key(provider: str) -> str:
    """Look up the API key for ``provider`` from the environment."""
    env_var = _ENV_VARS.get(provider)
    if env_var is None:
        raise SystemExit(f"No env-var mapping for provider {provider!r}")
    if provider == "huggingface_local":
        return os.environ.get(env_var, "unused")
    key = os.environ.get(env_var)
    if not key:
        raise SystemExit(f"Missing {env_var} for provider {provider!r}")
    return key


def _add_pdf_chunk_args(parser: argparse.ArgumentParser) -> None:
    """Shared ``--pdf`` / chunking arguments for both subcommands."""
    parser.add_argument("--pdf", required=True, help="Path to a PDF file.")
    parser.add_argument(
        "--chunk-size", type=int, default=2000, help="Chunk size (chars)."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200, help="Chunk overlap (chars)."
    )


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m rag.cli",
        description="Run RAG queries against a PDF.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable INFO-level logging."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    answer = sub.add_parser("answer", help="Run a single query through one retriever.")
    _add_pdf_chunk_args(answer)
    answer.add_argument("--query", required=True, help="Natural-language question.")
    answer.add_argument(
        "--retriever",
        default="reranker",
        choices=RETRIEVERS,
        help="Retrieval strategy (default: reranker).",
    )
    answer.add_argument(
        "--provider",
        default="groq",
        choices=SUPPORTED_PROVIDERS,
        help="LLM provider (default: groq).",
    )
    answer.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Provider-specific model id.",
    )
    answer.add_argument("--top-k", type=int, default=8, help="Documents to retrieve.")
    answer.add_argument(
        "--temperature", type=float, default=0.0, help="LLM sampling temperature."
    )
    answer.add_argument(
        "--show-contexts",
        action="store_true",
        help="Print the retrieved chunks before the answer.",
    )

    compare = sub.add_parser(
        "compare", help="Score all four retrievers on the same query."
    )
    _add_pdf_chunk_args(compare)
    compare.add_argument("--query", required=True, help="Natural-language question.")
    compare.add_argument(
        "--top-k", type=int, default=5, help="Cutoff for @k metrics."
    )
    compare.add_argument(
        "--gold-id",
        action="append",
        default=None,
        help=(
            "Hand-labeled relevant ``doc_id`` for the query. Repeat the flag "
            "to supply multiple ids. If omitted, the cross-encoder reranker "
            "is used as a stand-in oracle (note: this biases the metrics in "
            "favor of the reranker)."
        ),
    )

    return parser.parse_args(argv)


def _answer(args: argparse.Namespace) -> int:
    """Implement the ``answer`` subcommand."""
    docs = load_documents(args.pdf)
    chunks = chunk_documents(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    retriever = _build_retriever(args.retriever)
    retriever.add_documents(chunks)

    llm = LLM(api_key=_provider_api_key(args.provider)).get_llm(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
    )
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=args.top_k)
    response = pipeline.answer(args.query)

    if args.show_contexts:
        print(f"=== Retrieved contexts ({len(response.contexts)}) ===")
        for i, doc in enumerate(response.contexts, start=1):
            page = doc.metadata.get("page", "?")
            print(f"\n[Doc {i} | page={page}]")
            print(doc.page_content[:400], "..." if len(doc.page_content) > 400 else "")
        print()

    print("=== Answer ===")
    print(response.answer)

    contexts = [doc.page_content for doc in response.contexts]
    print(
        f"\nembedding_faithfulness = "
        f"{embedding_faithfulness(response.answer, contexts):.3f}"
    )
    return 0


def _compare(args: argparse.Namespace) -> int:
    """Implement the ``compare`` subcommand."""
    docs = load_documents(args.pdf)
    chunks = chunk_documents(
        docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    bm25 = BM25Retriever()
    dense = DenseRetriever()
    hybrid = HybridRetriever()
    reranker = RerankerRetriever(base_retriever=HybridRetriever(), candidate_pool=20)
    for retriever in (bm25, dense, hybrid, reranker):
        retriever.add_documents(chunks)

    if args.gold_id:
        gold_ids: List[str] = list(args.gold_id)
        print(f"Using hand-labeled gold ids: {gold_ids}")
    else:
        gold_ids = [hit.document.metadata["doc_id"] for hit in reranker.retrieve(args.query, k=3)]
        print(
            "WARNING: no --gold-id supplied; using cross-encoder top-3 as a stand-in "
            "oracle. The reranker will look artificially perfect."
        )
        print(f"Stand-in gold ids: {gold_ids}")

    dataset = [EvalExample(question=args.query, relevant_doc_ids=gold_ids)]

    print(f"\n=== Retrieval metrics @k={args.top_k} ===")
    for retriever_name, retriever in (
        ("bm25", bm25),
        ("dense", dense),
        ("hybrid", hybrid),
        ("reranker", reranker),
    ):
        report = evaluate_retriever(retriever, dataset, k=args.top_k)
        metrics = "  ".join(f"{k}={v:.3f}" for k, v in report.metrics.items())
        print(f"{retriever_name:<12} {metrics}")
    return 0


def main(argv=None) -> int:
    """CLI entry point."""
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.command == "answer":
        return _answer(args)
    if args.command == "compare":
        return _compare(args)
    raise SystemExit(f"Unknown command: {args.command}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
