"""
app/routers/ingest.py
──────────────────────
Ingestion endpoints:
  POST /ingest/file  — upload a PDF or text file
  POST /ingest/text  — ingest raw text directly
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.dependencies import IngestionDep, SettingsDep
from app.models.schemas import IngestResponse, IngestTextRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["Ingestion"])

SUPPORTED_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
}


@router.post(
    "/file",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF or text document",
)
async def ingest_file(
    settings: SettingsDep,
    ingestion: IngestionDep,
    file: UploadFile = File(..., description="PDF or text file to ingest"),
) -> IngestResponse:
    """
    Upload a file and ingest its content into the vector store.
    Re-uploading the same file replaces previous content (idempotent).
    """
    # Validate file size
    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb}MB",
        )

    # Validate content type
    content_type = file.content_type or ""
    filename = file.filename or "upload"

    if not any(t in content_type for t in SUPPORTED_TYPES) and not filename.endswith(
        tuple(settings.allowed_file_types)
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Allowed: {settings.allowed_file_types}",
        )

    try:
        result = await ingestion.ingest_pdf_bytes(
            file_bytes=file_bytes,
            filename=filename,
        )
    except Exception as exc:
        logger.error("Ingestion failed", extra={"filename": filename, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        ) from exc

    return IngestResponse(
        status="success",
        message=f"Successfully ingested {result['chunks_created']} chunks from {filename}",
        **result,
    )


@router.post(
    "/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest raw text content directly",
)
async def ingest_text(
    body: IngestTextRequest,
    ingestion: IngestionDep,
) -> IngestResponse:
    """Ingest arbitrary text without file upload (useful for API-driven pipelines)."""
    try:
        result = await ingestion.ingest_text(
            content=body.content,
            source_name=body.source_name,
            metadata=body.metadata,
        )
    except Exception as exc:
        logger.error("Text ingestion failed", extra={"source": body.source_name, "error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text ingestion failed: {exc}",
        ) from exc

    return IngestResponse(
        status="success",
        message=f"Successfully ingested {result['chunks_created']} chunks from '{body.source_name}'",
        **result,
    )
