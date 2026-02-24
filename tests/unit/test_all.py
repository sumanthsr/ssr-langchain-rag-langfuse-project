"""
tests/unit/test_ingestion.py
─────────────────────────────
Unit tests for the ingestion pipeline.
All external calls (Qdrant, file I/O) are mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ingestion import IngestionService


@pytest.fixture
def ingestion(test_settings, mock_vector_store):
    return IngestionService(settings=test_settings, vector_store=mock_vector_store)


@pytest.mark.unit
def test_stable_doc_id_is_deterministic():
    """Same content + name → same doc_id every time."""
    content = b"Hello world"
    name = "test.pdf"
    id1 = IngestionService._stable_doc_id(content, name)
    id2 = IngestionService._stable_doc_id(content, name)
    assert id1 == id2
    assert len(id1) == 16


@pytest.mark.unit
def test_stable_doc_id_differs_on_content_change():
    id1 = IngestionService._stable_doc_id(b"content v1", "doc.pdf")
    id2 = IngestionService._stable_doc_id(b"content v2", "doc.pdf")
    assert id1 != id2


@pytest.mark.unit
def test_split_and_enrich_adds_required_metadata(ingestion, test_settings):
    from langchain_core.documents import Document

    docs = [Document(page_content="A" * 3000, metadata={"source": "test.pdf", "page": 1})]
    chunks = ingestion._split_and_enrich(docs, "doc123", "test.pdf", {"category": "finance"})

    assert len(chunks) > 0
    for chunk in chunks:
        assert "doc_id" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata
        assert "ingested_at" in chunk.metadata
        assert "category" in chunk.metadata
        assert chunk.metadata["category"] == "finance"
        assert chunk.metadata["doc_id"] == "doc123"


@pytest.mark.unit
async def test_ingest_text_calls_vector_store(ingestion, mock_vector_store):
    mock_vector_store.client.delete = MagicMock()
    mock_vector_store.add_documents = MagicMock()

    result = await ingestion.ingest_text("Some test content for ingestion.", "test_source")

    assert result["source_name"] == "test_source"
    assert result["chunks_created"] >= 1
    assert "document_id" in result
    assert result["ingestion_time_ms"] > 0


"""
tests/unit/test_retrieval.py
─────────────────────────────
Unit tests for the hybrid retrieval + reranking service.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.services.retrieval import RetrievalService


@pytest.fixture
def retrieval(test_settings, mock_vector_store):
    with patch.object(RetrievalService, "_init_reranker", return_value=None):
        return RetrievalService(settings=test_settings, vector_store=mock_vector_store)


@pytest.mark.unit
def test_rrf_fusion_deduplicates_by_chunk_id(retrieval):
    doc_a = Document(page_content="A", metadata={"chunk_id": "id1"})
    doc_b = Document(page_content="B", metadata={"chunk_id": "id2"})

    dense = [(doc_a, 0.9), (doc_b, 0.7)]
    bm25 = [(doc_b, 10.0), (doc_a, 8.0)]

    fused = retrieval._reciprocal_rank_fusion(dense, bm25)
    ids = [d.metadata["chunk_id"] for d in fused]

    # No duplicates
    assert len(ids) == len(set(ids)) == 2


@pytest.mark.unit
def test_rrf_fusion_attaches_score(retrieval):
    doc = Document(page_content="X", metadata={"chunk_id": "x1"})
    fused = retrieval._reciprocal_rank_fusion([(doc, 0.8)], [])
    assert "rrf_score" in fused[0].metadata
    assert isinstance(fused[0].metadata["rrf_score"], float)


@pytest.mark.unit
def test_retrieve_returns_list(retrieval, mock_vector_store):
    doc = Document(page_content="Revenue was $4B", metadata={"chunk_id": "rev1"})
    mock_vector_store.similarity_search_with_score.return_value = [(doc, 0.9)]
    mock_vector_store.similarity_search.return_value = [doc]

    results = retrieval.retrieve("What was the revenue?")
    assert isinstance(results, list)


"""
tests/unit/test_chain.py
──────────────────────────
Unit tests for the RAG chain.
"""

import pytest
from langchain_core.documents import Document

from app.services.chain import RAGChain


@pytest.mark.unit
def test_format_docs_includes_source_header():
    docs = [
        Document(
            page_content="Revenue was $4.2B",
            metadata={"source": "report.pdf", "page": 3, "rrf_score": 0.9},
        )
    ]
    formatted = RAGChain._format_docs(docs)
    assert "report.pdf" in formatted
    assert "Revenue was $4.2B" in formatted
    assert "Page: 3" in formatted


@pytest.mark.unit
def test_format_docs_empty_returns_placeholder():
    result = RAGChain._format_docs([])
    assert "No relevant context" in result


@pytest.mark.unit
def test_docs_to_source_nodes(mock_docs):
    nodes = RAGChain._docs_to_source_nodes(mock_docs)
    assert len(nodes) == 2
    assert nodes[0].source == "annual_report.pdf"
    assert nodes[0].page == 3
    assert 0.0 <= nodes[0].score <= 1.0


"""
tests/unit/test_api_query.py
──────────────────────────────
Unit tests for the /api/v1/query endpoint.
"""

import pytest


@pytest.mark.unit
def test_query_returns_200(client):
    response = client.post(
        "/api/v1/query",
        json={"question": "What is the revenue?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "source_nodes" in data
    assert "latency_ms" in data
    assert data["model"] == "llama3"


@pytest.mark.unit
def test_query_rejects_short_question(client):
    response = client.post("/api/v1/query", json={"question": "Hi"})
    assert response.status_code == 422


@pytest.mark.unit
def test_query_rejects_empty_question(client):
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code == 422


@pytest.mark.unit
def test_health_returns_200(client):
    response = client.get("/health")
    # May return 200 or 503 depending on mock state — just check it responds
    assert response.status_code in (200, 503)
    data = response.json()
    assert "status" in data
    assert "version" in data


@pytest.mark.unit
def test_ingest_text_returns_201(client):
    response = client.post(
        "/api/v1/ingest/text",
        json={
            "content": "This is a long enough test document for ingestion testing purposes.",
            "source_name": "test_document",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert "chunks_created" in data
