"""
app/routers/query.py
─────────────────────
Query endpoints:
  POST /query        — single shot Q&A
  GET  /query/stream — streaming Q&A (SSE)

Every request is traced in Langfuse with:
  - Full question text
  - Retrieved source nodes
  - LLM model + latency
  - Cache hit/miss status
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, AsyncIterator
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.dependencies import ChainDep, ObsDep, RedisDep, RetrievalDep, SettingsDep
from app.models.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


def _cache_key(question: str, filters: dict[str, Any] | None) -> str:
    """Stable cache key from question + filters."""
    payload = json.dumps({"q": question.strip().lower(), "f": filters or {}}, sort_keys=True)
    return f"rag:cache:{hashlib.sha256(payload.encode()).hexdigest()}"


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question over the ingested document corpus",
    responses={
        200: {"description": "Answer generated successfully"},
        422: {"description": "Invalid request payload"},
        503: {"description": "LLM or vector store unavailable"},
    },
)
async def query(
    body: QueryRequest,
    settings: SettingsDep,
    retrieval: RetrievalDep,
    chain: ChainDep,
    obs: ObsDep,
    redis: RedisDep,
) -> QueryResponse:
    """
    Full RAG pipeline:
      1. Check Redis cache (skip LLM if cached)
      2. Hybrid retrieve + rerank
      3. LLM generation (traced by Langfuse)
      4. Cache response
      5. Return answer + source citations
    """
    request_start = time.perf_counter()
    session_id_str = str(body.session_id)
    trace_id: str | None = None

    # ── Step 1: Cache check ──────────────────────────────────────────────
    cache_key = _cache_key(body.question, body.filters)
    if redis and settings.cache_enabled:
        cached = redis.get(cache_key)
        if cached:
            logger.info("Cache hit", extra={"session_id": session_id_str})
            cached_data = json.loads(cached)
            cached_data["latency_ms"] = (time.perf_counter() - request_start) * 1000
            return QueryResponse(**cached_data)

    # ── Step 2: Retrieval ────────────────────────────────────────────────
    try:
        docs = retrieval.retrieve(
            query=body.question,
            top_k=body.top_k,
            filters=body.filters,
        )
    except Exception as exc:
        logger.error("Retrieval failed", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval failed: {exc}",
        ) from exc

    # ── Step 3: LLM generation with Langfuse tracing ─────────────────────
    try:
        with obs.trace_context(
            name="rag-query",
            session_id=session_id_str,
            metadata={
                "question": body.question,
                "retrieved_chunks": len(docs),
                "top_k": body.top_k,
                "filters": body.filters,
            },
        ) as (trace, handler):
            trace_id = getattr(trace, "id", None)

            result = await chain.ainvoke(
                question=body.question,
                docs=docs,
                callbacks=[handler],
            )
    except Exception as exc:
        logger.error("LLM chain failed", extra={"error": str(exc), "session": session_id_str})
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM generation failed: {exc}",
        ) from exc

    total_latency_ms = (time.perf_counter() - request_start) * 1000

    # ── Step 4: Build response ───────────────────────────────────────────
    response = QueryResponse(
        answer=result["answer"],
        session_id=body.session_id,
        trace_id=str(trace_id) if trace_id else None,
        source_nodes=result["source_nodes"],
        latency_ms=total_latency_ms,
        model=result["model"],
    )

    # ── Step 5: Cache the response ───────────────────────────────────────
    if redis and settings.cache_enabled:
        try:
            redis.setex(
                cache_key,
                settings.redis_cache_ttl,
                response.model_dump_json(exclude={"latency_ms", "created_at", "trace_id"}),
            )
        except Exception as exc:
            logger.warning("Cache write failed", extra={"error": str(exc)})

    logger.info(
        "Query complete",
        extra={
            "latency_ms": round(total_latency_ms, 1),
            "sources": len(response.source_nodes),
            "session_id": session_id_str,
            "trace_id": trace_id,
        },
    )
    return response


@router.get(
    "/stream",
    summary="Stream an answer token-by-token (Server-Sent Events)",
    response_class=StreamingResponse,
)
async def query_stream(
    question: str,
    session_id: UUID | None = None,
    retrieval: RetrievalDep = None,
    chain: ChainDep = None,
    obs: ObsDep = None,
) -> StreamingResponse:
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Tokens arrive in real-time as the LLM generates them.
    Use EventSource in the browser or httpx.stream() in Python clients.
    """

    async def _generate() -> AsyncIterator[str]:
        docs = retrieval.retrieve(query=question)
        trace = obs.trace(name="rag-stream", session_id=str(session_id) if session_id else None)
        handler = obs.callback_handler(trace_id=getattr(trace, "id", None))

        try:
            async for token in chain.astream(
                question=question, docs=docs, callbacks=[handler]
            ):
                # SSE format: data: <payload>\n\n
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        finally:
            obs.flush()

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable Nginx buffering for SSE
        },
    )
