"""End-to-end RAG pipeline: retrieve, format context, ask the LLM.

The pipeline is deliberately retriever- and LLM-agnostic. Anything satisfying
:class:`rag.retrievers.base.BaseRetriever` works as a retriever; anything with
either an ``invoke(str | dict) -> AIMessage | str`` method or that is itself
callable works as an LLM. This keeps the pipeline compatible with both
LangChain chat models and the raw HuggingFace adapters in
:mod:`rag.huggingface_llm`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from rag.prompts import default_prompt
from rag.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """The output of :meth:`RAGPipeline.answer`.

    Attributes:
        question: The original question.
        answer: The LLM-generated answer text.
        contexts: Retrieved context documents shown to the LLM.
    """

    question: str
    answer: str
    contexts: List[Document]


def format_context(documents: List[Document]) -> str:
    """Concatenate documents into a single context string with source tags.

    Args:
        documents: Retrieved documents.

    Returns:
        A newline-delimited string with one block per document, each tagged
        with its source metadata when present.
    """
    blocks: List[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source") if doc.metadata else None
        page = doc.metadata.get("page") if doc.metadata else None
        header = f"[Doc {i}"
        if source:
            header += f" | source={source}"
        if page is not None:
            header += f" | page={page}"
        header += "]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def _invoke_llm(llm: Any, prompt: str) -> str:
    """Call ``llm`` with ``prompt`` and return the text answer.

    Supports three calling conventions, in order of preference:

    1. LangChain ``Runnable.invoke(prompt)`` returning an ``AIMessage`` whose
       ``.content`` holds the text.
    2. A bare ``invoke(prompt)`` returning a string.
    3. A callable ``llm(prompt)`` returning a string.
    """
    if hasattr(llm, "invoke"):
        result = llm.invoke(prompt)
        return getattr(result, "content", result) if not isinstance(result, str) else result
    if callable(llm):
        return llm(prompt)
    raise TypeError(f"Unsupported LLM object: {type(llm)!r}")


class RAGPipeline:
    """Compose a retriever, prompt, and LLM into an answerable pipeline."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        prompt: Optional[PromptTemplate] = None,
        top_k: int = 5,
    ) -> None:
        """Build the pipeline.

        Args:
            retriever: Any :class:`BaseRetriever` implementation.
            llm: A LangChain chat model, a HuggingFace adapter, or any object
                exposing ``invoke(str)`` / ``__call__(str)``.
            prompt: Prompt template with ``context`` and ``question`` inputs.
                Defaults to :func:`rag.prompts.default_prompt`.
            top_k: Number of context documents to retrieve per question.

        Raises:
            ValueError: If ``top_k`` is not positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._retriever = retriever
        self._llm = llm
        self._prompt = prompt or default_prompt()
        self._top_k = top_k

    @property
    def retriever(self) -> BaseRetriever:
        """The configured retriever."""
        return self._retriever

    def answer(self, question: str) -> RAGResponse:
        """Run retrieval and generation for a question.

        Args:
            question: Natural-language question.

        Returns:
            A :class:`RAGResponse` with the answer and supporting contexts.
        """
        contexts = self._retriever.retrieve_documents(question, k=self._top_k)
        prompt_str = self._prompt.format(
            question=question,
            context=format_context(contexts),
        )
        answer_text = _invoke_llm(self._llm, prompt_str)
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)
        return RAGResponse(question=question, answer=answer_text, contexts=contexts)
