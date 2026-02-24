"""
app/models/schemas.py
─────────────────────
Pydantic v2 request and response schemas for all API endpoints.
Strict typing ensures invalid payloads are rejected before reaching business logic.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ── Shared ─────────────────────────────────────────────────────────────────

class SourceNode(BaseModel):
    """A single retrieved document chunk that contributed to the answer."""

    chunk_id: str = Field(..., description="Unique ID of this chunk in the vector store")
    content: str = Field(..., description="The raw text of this chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0–1)")
    source: str = Field(..., description="Source document filename or URL")
    page: int | None = Field(None, description="Page number within the source document")
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Query endpoint ──────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The user's question to answer from the document corpus",
        examples=["What are the main revenue drivers mentioned in the report?"],
    )
    session_id: UUID = Field(
        default_factory=uuid4,
        description="Session ID for conversation continuity and Langfuse tracing",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Override default number of chunks to retrieve (optional)",
    )
    filters: dict[str, Any] | None = Field(
        None,
        description="Optional metadata filters for scoped retrieval (e.g. source file)",
        examples=[{"source": "annual_report_2024.pdf"}],
    )
    stream: bool = Field(False, description="If true, use GET /query/stream instead")

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class QueryResponse(BaseModel):
    """Response body for POST /query"""

    answer: str = Field(..., description="The LLM-generated answer")
    session_id: UUID
    trace_id: str | None = Field(None, description="Langfuse trace ID for this query")
    source_nodes: list[SourceNode] = Field(
        default_factory=list,
        description="The document chunks used to generate the answer",
    )
    latency_ms: float = Field(..., description="Total query latency in milliseconds")
    model: str = Field(..., description="LLM model used to generate the answer")
    tokens_used: int | None = Field(None, description="Approximate token count")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ── Ingest endpoints ────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    """Request body for POST /ingest/text — ingest raw text directly"""

    content: str = Field(..., min_length=10, description="Raw text content to ingest")
    source_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="A human-readable name for this document",
        examples=["Q4 2024 Earnings Call Transcript"],
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to attach to all chunks from this document",
    )


class IngestResponse(BaseModel):
    """Response body for all ingest endpoints"""

    status: str = Field(..., examples=["success"])
    document_id: str = Field(..., description="Unique ID assigned to this document")
    chunks_created: int = Field(..., description="Number of chunks stored in the vector DB")
    source_name: str
    ingestion_time_ms: float
    message: str = Field(..., description="Human-readable status message")


# ── Health endpoints ────────────────────────────────────────────────────────

class DependencyStatus(BaseModel):
    name: str
    status: str       # "ok" | "degraded" | "down"
    latency_ms: float | None = None
    detail: str | None = None


class HealthResponse(BaseModel):
    """Response body for GET /health"""

    status: str       # "ok" | "degraded" | "down"
    version: str
    environment: str
    uptime_seconds: float
    dependencies: list[DependencyStatus] = Field(default_factory=list)


class ReadinessResponse(BaseModel):
    """Response body for GET /ready (used by K8s readiness probe)"""

    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)


# ── Error responses ─────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Standard error envelope returned for all 4xx/5xx responses"""

    error: str
    details: list[ErrorDetail] = Field(default_factory=list)
    trace_id: str | None = None
    request_id: str | None = None
