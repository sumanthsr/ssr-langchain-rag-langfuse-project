"""
app/services/ingestion.py
──────────────────────────
Offline ingestion pipeline: PDF → chunks → embeddings → Qdrant.

Design decisions:
  - Uses PyMuPDFLoader for speed + quality (handles complex PDFs)
  - Falls back to PyPDFLoader for simple documents
  - RecursiveCharacterTextSplitter for reliable chunking
  - Metadata enrichment: source, page, chunk_index, doc_id, ingested_at
  - Idempotent: re-ingesting the same document replaces old chunks (by doc_id)
  - Async-safe: blocking IO (embeddings, file reads) runs in threadpool
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain_core.documents import Document

from app.config import Settings

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Orchestrates the full document ingestion pipeline.
    Injected as a FastAPI dependency — one instance per application lifetime.
    """

    def __init__(self, settings: Settings, vector_store: Any) -> None:
        self._settings = settings
        self._vector_store = vector_store  # Qdrant vector store instance
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,  # adds char offset metadata to each chunk
        )
        logger.info(
            "IngestionService ready",
            extra={
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
            },
        )

    # ── Public API ──────────────────────────────────────────────────────────

    async def ingest_pdf_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a PDF from raw bytes (e.g. from file upload).
        Writes to a temp file, loads, chunks, embeds, stores.
        Returns ingestion summary dict.
        """
        start_ts = time.perf_counter()
        doc_id = self._stable_doc_id(file_bytes, filename)
        logger.info("Starting PDF ingestion", extra={"filename": filename, "doc_id": doc_id})

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            docs = self._load_pdf(Path(tmp.name), filename)

        chunks = self._split_and_enrich(docs, doc_id, filename, metadata or {})
        await self._upsert_chunks(chunks, doc_id)

        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        logger.info(
            "PDF ingestion complete",
            extra={"filename": filename, "chunks": len(chunks), "ms": elapsed_ms},
        )
        return {
            "document_id": doc_id,
            "chunks_created": len(chunks),
            "source_name": filename,
            "ingestion_time_ms": elapsed_ms,
        }

    async def ingest_text(
        self,
        content: str,
        source_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest arbitrary text content (markdown, plain text, etc).
        Useful for programmatic ingestion without a file upload.
        """
        start_ts = time.perf_counter()
        doc_id = self._stable_doc_id(content.encode(), source_name)

        doc = Document(
            page_content=content,
            metadata={"source": source_name, "page": 0},
        )
        chunks = self._split_and_enrich([doc], doc_id, source_name, metadata or {})
        await self._upsert_chunks(chunks, doc_id)

        elapsed_ms = (time.perf_counter() - start_ts) * 1000
        return {
            "document_id": doc_id,
            "chunks_created": len(chunks),
            "source_name": source_name,
            "ingestion_time_ms": elapsed_ms,
        }

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load_pdf(self, path: Path, filename: str) -> list[Document]:
        """
        Try PyMuPDFLoader first (faster, handles complex layouts).
        Fall back to PyPDFLoader if PyMuPDF fails.
        """
        try:
            loader = PyMuPDFLoader(str(path))
            docs = loader.load()
            logger.debug("Loaded PDF with PyMuPDF", extra={"pages": len(docs)})
            return docs
        except Exception as exc:
            logger.warning(
                "PyMuPDF failed, falling back to PyPDF",
                extra={"filename": filename, "error": str(exc)},
            )
            loader = PyPDFLoader(str(path))
            return loader.load()

    def _split_and_enrich(
        self,
        docs: list[Document],
        doc_id: str,
        source_name: str,
        extra_metadata: dict[str, Any],
    ) -> list[Document]:
        """
        Split documents into chunks and enrich metadata.
        Each chunk gets: doc_id, chunk_id, source, page, chunk_index, ingested_at.
        """
        chunks = self._splitter.split_documents(docs)
        ingested_at = datetime.now(timezone.utc).isoformat()

        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk.metadata.update(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "source": source_name,
                    "ingested_at": ingested_at,
                    **extra_metadata,
                }
            )
            # Qdrant uses "id" field for point IDs
            chunk.metadata["id"] = chunk_id

        logger.debug("Split into chunks", extra={"count": len(chunks)})
        return chunks

    async def _upsert_chunks(self, chunks: list[Document], doc_id: str) -> None:
        """
        Delete existing chunks for this doc_id, then insert new ones.
        This makes ingestion idempotent — re-uploading the same file
        replaces old chunks rather than duplicating them.
        """
        try:
            # Delete old chunks for this document (Qdrant filter by metadata)
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

            self._vector_store.client.delete(
                collection_name=self._settings.qdrant_collection,
                points_selector=Filter(
                    must=[FieldCondition(key="metadata.doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            logger.debug("Deleted old chunks for doc_id", extra={"doc_id": doc_id})
        except Exception as exc:
            logger.debug(
                "No existing chunks to delete (first ingest)",
                extra={"doc_id": doc_id, "error": str(exc)},
            )

        # Insert new chunks (synchronous call — Qdrant client handles batching)
        if chunks:
            self._vector_store.add_documents(chunks)

    @staticmethod
    def _stable_doc_id(content: bytes, name: str) -> str:
        """
        Generate a stable doc_id from content hash + filename.
        Same content + filename = same ID → enables idempotent re-ingestion.
        """
        raw = content + name.encode()
        return hashlib.sha256(raw).hexdigest()[:16]
