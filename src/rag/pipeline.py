"""RAG pipeline combining retrieval and generation."""

from __future__ import annotations

from dataclasses import dataclass

from rag.llm import LLM
from rag.retrievers import Retriever


@dataclass
class RAGResult:
    """Result from RAG pipeline."""

    answer: str
    contexts: list[str]
    doc_ids: list[str]


class RAGPipeline:
    """RAG system combining retriever and LLM."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        top_k: int = 5,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer(self, query: str) -> RAGResult:
        """Answer a question using retrieval and generation."""
        docs = self.retriever.retrieve(query, top_k=self.top_k)
        contexts = [doc.page_content for doc in docs]
        doc_ids = [doc.metadata.get("doc_id", "") for doc in docs]

        context_text = "\n\n".join(
            [f"[{doc_id}] {ctx}" for doc_id, ctx in zip(doc_ids, contexts)]
        )
        prompt = f"""Answer the question based on the provided context.

Context:
{context_text}

Question: {query}

Answer:"""

        answer = self.llm.generate(prompt)
        return RAGResult(answer=answer, contexts=contexts, doc_ids=doc_ids)
