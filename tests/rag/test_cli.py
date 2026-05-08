"""Tests for ``rag.cli`` argument parsing and dispatch.

End-to-end CLI runs hit real models and APIs, so we test argparse + the
retriever-builder helpers directly rather than running ``main``.
"""
from __future__ import annotations

import pytest

from rag.cli import RETRIEVERS, _build_retriever, parse_args
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankerRetriever,
)


class TestBuildRetriever:
    def test_returns_correct_types(self):
        assert isinstance(_build_retriever("bm25"), BM25Retriever)
        assert isinstance(_build_retriever("dense"), DenseRetriever)
        assert isinstance(_build_retriever("hybrid"), HybridRetriever)
        assert isinstance(_build_retriever("reranker"), RerankerRetriever)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            _build_retriever("colbert")

    def test_retriever_names_match_choices(self):
        assert set(RETRIEVERS) == {"bm25", "dense", "hybrid", "reranker"}


class TestParseArgs:
    def test_answer_defaults(self):
        args = parse_args(
            ["answer", "--pdf", "x.pdf", "--query", "what?"]
        )
        assert args.command == "answer"
        assert args.retriever == "reranker"
        assert args.provider == "groq"
        assert args.top_k == 8

    def test_answer_overrides(self):
        args = parse_args(
            [
                "answer",
                "--pdf", "x.pdf",
                "--query", "q",
                "--retriever", "bm25",
                "--provider", "openai",
                "--model", "gpt-4o-mini",
                "--top-k", "3",
                "--show-contexts",
            ]
        )
        assert args.retriever == "bm25"
        assert args.provider == "openai"
        assert args.top_k == 3
        assert args.show_contexts is True

    def test_compare_defaults(self):
        args = parse_args(["compare", "--pdf", "x.pdf", "--query", "q"])
        assert args.command == "compare"
        assert args.gold_id is None
        assert args.top_k == 5

    def test_compare_with_gold_ids(self):
        args = parse_args(
            [
                "compare",
                "--pdf", "x.pdf",
                "--query", "q",
                "--gold-id", "chunk_1",
                "--gold-id", "chunk_42",
            ]
        )
        assert args.gold_id == ["chunk_1", "chunk_42"]

    def test_subcommand_required(self):
        with pytest.raises(SystemExit):
            parse_args([])
