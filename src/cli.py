"""CLI for RAG system."""

from __future__ import annotations

import argparse
import os

from rag import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RAGPipeline,
    RerankerRetriever,
    chunk_documents,
    get_llm,
    load_documents,
)


def get_retriever(retriever_type: str):
    """Get retriever by name."""
    if retriever_type == "bm25":
        return BM25Retriever()
    elif retriever_type == "dense":
        return DenseRetriever()
    elif retriever_type == "hybrid":
        return HybridRetriever()
    elif retriever_type == "reranker":
        base = HybridRetriever()
        return RerankerRetriever(base_retriever=base)
    else:
        raise ValueError(f"Unknown retriever: {retriever_type}")


def answer_command(args):
    """Answer a single query."""
    docs = load_documents(args.pdf)
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)

    retriever = get_retriever(args.retriever)
    retriever.add_documents(chunks)

    llm = get_llm(
        provider=args.provider,
        model=args.model,
        api_key=os.getenv(f"{args.provider.upper()}_API_KEY"),
    )
    pipeline = RAGPipeline(retriever=retriever, llm=llm, top_k=args.top_k)

    result = pipeline.answer(args.query)

    if args.show_contexts:
        print("=" * 80)
        print("RETRIEVED CONTEXTS:")
        print("=" * 80)
        for doc_id, context in zip(result.doc_ids, result.contexts):
            print(f"\n[{doc_id}]")
            print(context[:500])

    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result.answer)


def compare_command(args):
    """Compare all retrievers on same query."""
    docs = load_documents(args.pdf)
    chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=100)

    from rag.metrics import hit_rate_at_k, precision_at_k, recall_at_k

    gold_ids = args.gold_id if args.gold_id else None

    print(f"\nQuery: {args.query}\n")
    print("Retriever Results:")
    print("-" * 80)

    for retriever_name in ["bm25", "dense", "hybrid", "reranker"]:
        retriever = get_retriever(retriever_name)
        retriever.add_documents(chunks)

        results = retriever.retrieve(args.query, top_k=args.top_k)
        retrieved_ids = [doc.metadata.get("doc_id", "") for doc in results]

        if gold_ids:
            hit = hit_rate_at_k(retrieved_ids, gold_ids, k=args.top_k)
            prec = precision_at_k(retrieved_ids, gold_ids, k=args.top_k)
            rec = recall_at_k(retrieved_ids, gold_ids, k=args.top_k)
            print(
                f"{retriever_name:12} | Hit@{args.top_k}: {hit:.2f} | "
                f"Precision@{args.top_k}: {prec:.2f} | Recall@{args.top_k}: {rec:.2f}"
            )
        else:
            print(f"{retriever_name:12} | Retrieved: {', '.join(retrieved_ids)}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG system for SEC filings")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    answer_parser = subparsers.add_parser("answer", help="Answer a single query")
    answer_parser.add_argument("--pdf", required=True, help="Path to PDF")
    answer_parser.add_argument(
        "--retriever",
        default="bm25",
        choices=["bm25", "dense", "hybrid", "reranker"],
        help="Retriever type",
    )
    answer_parser.add_argument("--provider", default="groq", help="LLM provider")
    answer_parser.add_argument("--model", help="Model name")
    answer_parser.add_argument("--query", required=True, help="Query text")
    answer_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )
    answer_parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="Show retrieved contexts",
    )
    answer_parser.set_defaults(func=answer_command)

    compare_parser = subparsers.add_parser(
        "compare", help="Compare all retrievers on same query"
    )
    compare_parser.add_argument("--pdf", required=True, help="Path to PDF")
    compare_parser.add_argument("--query", required=True, help="Query text")
    compare_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of documents to retrieve"
    )
    compare_parser.add_argument(
        "--gold-id",
        action="append",
        help="Gold document IDs for evaluation (repeatable)",
    )
    compare_parser.set_defaults(func=compare_command)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
