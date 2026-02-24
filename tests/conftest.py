"""
tests/conftest.py
──────────────────
Shared pytest fixtures for unit and integration tests.
Unit tests mock all external dependencies.
Integration tests use testcontainers for real services.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.config import Settings
from app.main import create_app
from app.models.schemas import SourceNode


# ── Test settings ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for testing — no real services needed."""
    return Settings(
        environment="dev",
        debug=True,
        log_level="DEBUG",
        qdrant_host="localhost",
        qdrant_port=6333,
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3",
        redis_url="redis://localhost:6379/0",
        cache_enabled=False,       # disable caching in unit tests
        langfuse_enabled=False,    # disable observability in unit tests
        reranker_enabled=False,    # disable reranker in unit tests (slow)
    )


# ── Mock fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def mock_docs() -> list[Any]:
    """Sample Document objects for use in tests."""
    from langchain_core.documents import Document

    return [
        Document(
            page_content="The company reported revenue of $4.2 billion in Q3 2024.",
            metadata={
                "source": "annual_report.pdf",
                "page": 3,
                "chunk_id": "abc123",
                "rrf_score": 0.85,
            },
        ),
        Document(
            page_content="Operating expenses increased by 12% year-over-year.",
            metadata={
                "source": "annual_report.pdf",
                "page": 4,
                "chunk_id": "def456",
                "rrf_score": 0.72,
            },
        ),
    ]


@pytest.fixture
def mock_source_nodes() -> list[SourceNode]:
    return [
        SourceNode(
            chunk_id="abc123",
            content="The company reported revenue of $4.2 billion in Q3 2024.",
            score=0.85,
            source="annual_report.pdf",
            page=3,
        )
    ]


@pytest.fixture
def mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.similarity_search_with_score.return_value = []
    vs.similarity_search.return_value = []
    vs.add_documents.return_value = None
    vs.client = MagicMock()
    vs.client.get_collections.return_value = MagicMock(collections=[])
    return vs


@pytest.fixture
def mock_retrieval_service(mock_docs: list[Any]) -> MagicMock:
    svc = MagicMock()
    svc.retrieve.return_value = mock_docs
    return svc


@pytest.fixture
def mock_ingestion_service() -> AsyncMock:
    svc = AsyncMock()
    svc.ingest_pdf_bytes.return_value = {
        "document_id": "test_doc_001",
        "chunks_created": 5,
        "source_name": "test.pdf",
        "ingestion_time_ms": 123.4,
    }
    svc.ingest_text.return_value = {
        "document_id": "test_text_001",
        "chunks_created": 3,
        "source_name": "test content",
        "ingestion_time_ms": 45.6,
    }
    return svc


@pytest.fixture
def mock_rag_chain(mock_source_nodes: list[SourceNode]) -> AsyncMock:
    chain = AsyncMock()
    chain.ainvoke.return_value = {
        "answer": "The company reported revenue of $4.2 billion in Q3 2024.",
        "source_nodes": mock_source_nodes,
        "latency_ms": 1234.5,
        "model": "llama3",
    }
    return chain


@pytest.fixture
def mock_obs_client() -> MagicMock:
    obs = MagicMock()
    obs.enabled = False
    obs.trace.return_value = MagicMock(id="test-trace-id")
    obs.callback_handler.return_value = MagicMock()
    obs.flush.return_value = None
    # Make trace_context work as a context manager
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=(MagicMock(id="trace-id"), MagicMock()))
    ctx.__exit__ = MagicMock(return_value=False)
    obs.trace_context.return_value = ctx
    return obs


# ── App fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def test_app(
    test_settings: Settings,
    mock_vector_store: MagicMock,
    mock_retrieval_service: MagicMock,
    mock_ingestion_service: AsyncMock,
    mock_rag_chain: AsyncMock,
    mock_obs_client: MagicMock,
) -> Any:
    """
    FastAPI app with all dependencies mocked.
    Bypasses lifespan (no real service connections in unit tests).
    """
    import app.dependencies as deps

    deps._vector_store = mock_vector_store
    deps._obs_client = mock_obs_client
    deps._ingestion_service = mock_ingestion_service
    deps._retrieval_service = mock_retrieval_service
    deps._rag_chain = mock_rag_chain
    deps._redis_client = None

    with patch("app.dependencies.get_settings", return_value=test_settings):
        application = create_app()

    return application


@pytest.fixture
def client(test_app: Any) -> TestClient:
    """Synchronous test client (for simple endpoint tests)."""
    return TestClient(test_app, raise_server_exceptions=True)


@pytest_asyncio.fixture
async def async_client(test_app: Any) -> AsyncIterator[AsyncClient]:
    """Async test client for async endpoint tests."""
    async with AsyncClient(app=test_app, base_url="http://test") as ac:
        yield ac
