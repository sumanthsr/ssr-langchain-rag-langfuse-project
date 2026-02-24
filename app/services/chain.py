"""
app/services/chain.py
──────────────────────
LCEL RAG chain with full Langfuse observability.

Architecture:
  retriever | format_docs | prompt | llm | StrOutputParser

Every LLM call is automatically traced by the Langfuse callback handler
passed via chain.ainvoke(..., config={"callbacks": [handler]}).
Langfuse captures: full prompt, response, token counts, latency per step.

Uses Ollama (local, open-source) as the LLM backend.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import Settings
from app.models.schemas import SourceNode

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────
# Kept concise and explicit. Critical rules:
#   1. Answer ONLY from the context provided
#   2. Cite sources (tells LLM to reference the source metadata)
#   3. Say "I don't know" rather than hallucinate
SYSTEM_PROMPT = """You are a precise question-answering assistant.
Your answers are grounded ONLY in the context passages provided below.

Rules:
1. Use ONLY information from the context to answer. Do not use prior knowledge.
2. If the context does not contain the answer, respond exactly: "I don't have enough information in the provided documents to answer this question."
3. Cite the source document name and page number when available, e.g. (annual_report.pdf, page 4).
4. Be concise. Do not pad your answer with unnecessary phrases.
5. If the question asks for a list, use bullet points.

Context:
{context}"""

HUMAN_TEMPLATE = "Question: {question}"


class RAGChain:
    """
    Encapsulates the full RAG generation chain.
    Injected as a FastAPI dependency — single instance per app lifetime.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = self._build_llm()
        self._chain = self._build_chain()
        logger.info(
            "RAGChain ready",
            extra={"model": settings.ollama_model, "url": settings.ollama_base_url},
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def ainvoke(
        self,
        question: str,
        docs: list[Document],
        callbacks: list[Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run the RAG chain asynchronously.
        Returns answer text + structured source nodes.
        """
        start_ts = time.perf_counter()
        context_text = self._format_docs(docs)

        try:
            answer = await self._chain.ainvoke(
                {"context": context_text, "question": question},
                config={"callbacks": callbacks or []},
            )
        except Exception as exc:
            logger.error("LLM chain error", extra={"error": str(exc)})
            raise

        latency_ms = (time.perf_counter() - start_ts) * 1000
        source_nodes = self._docs_to_source_nodes(docs)

        logger.info(
            "Chain invocation complete",
            extra={"latency_ms": latency_ms, "sources": len(source_nodes)},
        )
        return {
            "answer": answer,
            "source_nodes": source_nodes,
            "latency_ms": latency_ms,
            "model": self._settings.ollama_model,
        }

    async def astream(
        self,
        question: str,
        docs: list[Document],
        callbacks: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the LLM response token-by-token.
        Use with FastAPI StreamingResponse for real-time UX.
        """
        context_text = self._format_docs(docs)
        async for chunk in self._chain.astream(
            {"context": context_text, "question": question},
            config={"callbacks": callbacks or []},
        ):
            yield chunk

    # ── Chain construction ───────────────────────────────────────────────────

    def _build_llm(self) -> ChatOllama:
        """
        ChatOllama — talks to local Ollama server.
        temperature=0.0 for deterministic, factual RAG answers.
        """
        return ChatOllama(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_model,
            temperature=self._settings.ollama_temperature,
            num_ctx=self._settings.ollama_num_ctx,
            timeout=self._settings.ollama_timeout,
        )

    def _build_chain(self) -> Any:
        """
        LCEL pipeline using | operator.
        RunnablePassthrough preserves the question through the context step.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                ("human", HUMAN_TEMPLATE),
            ]
        )
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | self._llm
            | StrOutputParser()
        )

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        """
        Format retrieved documents into context string.
        Each chunk gets a header with source + page for citation.
        """
        if not docs:
            return "No relevant context found."

        parts = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("reranker_score", doc.metadata.get("rrf_score", "?"))
            header = f"[{i}] Source: {source} | Page: {page} | Score: {score}"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _docs_to_source_nodes(docs: list[Document]) -> list[SourceNode]:
        """Convert retrieved Documents to SourceNode schema objects."""
        nodes = []
        for doc in docs:
            score = doc.metadata.get("reranker_score") or doc.metadata.get("rrf_score") or 0.0
            nodes.append(
                SourceNode(
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    content=doc.page_content[:500],  # truncate for response size
                    score=min(max(float(score), 0.0), 1.0),
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    metadata={
                        k: v
                        for k, v in doc.metadata.items()
                        if k not in {"chunk_id", "source", "page"}
                    },
                )
            )
        return nodes
