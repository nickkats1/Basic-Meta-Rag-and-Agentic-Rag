"""Prompt templates used by the RAG pipeline.

Centralizing prompts here keeps the wording in one place so that A/B testing
prompt variants does not require touching pipeline or retriever code.
"""
from __future__ import annotations

from langchain_core.prompts import PromptTemplate


DEFAULT_RAG_TEMPLATE = """You are a careful financial analyst answering questions about SEC filings.

Use ONLY the provided context. If the context does not contain the answer, reply
exactly: "I don't know based on the provided context." Do not invent figures,
names, or dates. Quote numbers verbatim when possible and cite the source page
in parentheses when page metadata is available.

Context:
{context}

Question:
{question}

Answer:"""


CONCISE_RAG_TEMPLATE = """Answer the question using only the context. If the
context is insufficient, say "I don't know.".

Context:
{context}

Question: {question}
Answer:"""


def default_prompt() -> PromptTemplate:
    """Return the default RAG ``PromptTemplate``."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=DEFAULT_RAG_TEMPLATE,
    )


def concise_prompt() -> PromptTemplate:
    """Return a shorter RAG ``PromptTemplate`` suitable for small models."""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=CONCISE_RAG_TEMPLATE,
    )


__all__ = [
    "CONCISE_RAG_TEMPLATE",
    "DEFAULT_RAG_TEMPLATE",
    "concise_prompt",
    "default_prompt",
]
