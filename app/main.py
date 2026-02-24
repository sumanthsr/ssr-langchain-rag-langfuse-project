"""
app/main.py
────────────
FastAPI application entrypoint.

Lifespan pattern (FastAPI >= 0.93):
  - startup: initialise all services (embeddings, Qdrant, Langfuse, Redis)
  - shutdown: flush Langfuse, close Redis, release resources

Middleware:
  - CORS
  - Request ID injection (X-Request-ID header)
  - Structured request logging
  - Global exception handler → consistent error envelope

Routers mounted at /api/v1.
OpenAPI docs at /docs (disabled in production).
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.dependencies import init_all, shutdown_all
from app.models.schemas import ErrorResponse
from app.routers import health, ingest, query

# ── Structured logging setup ──────────────────────────────────────────────
def configure_logging(log_level: str) -> None:
    """Configure structlog for JSON output in prod, pretty in dev."""
    import sys

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
            if log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level, logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    # Also configure stdlib logging to route through structlog
    logging.basicConfig(level=log_level, handlers=[logging.StreamHandler()])


logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifecycle manager.
    All services are initialised here — once per process, not per request.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    logger.info(
        "Starting RAG service",
        extra={"env": settings.environment, "version": settings.app_version},
    )

    try:
        init_all(settings)
        logger.info("All services ready — accepting requests")
        yield
    finally:
        logger.info("Shutting down RAG service")
        shutdown_all()


# ── App factory ───────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="RAG API — LangChain",
        description=(
            "Production-ready RAG pipeline: PDF ingestion → hybrid retrieval → "
            "LLM answer generation. 100% open-source stack."
        ),
        version=settings.app_version,
        lifespan=lifespan,
        # Disable interactive docs in production
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID + structured logging middleware ─────────────────────
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):  # type: ignore
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(elapsed, 1),
                "request_id": request_id,
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round(elapsed, 1))
        return response

    # ── Global exception handler ──────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", "unknown")
        logger.exception(
            "Unhandled exception",
            extra={"path": request.url.path, "request_id": request_id},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                request_id=request_id,
            ).model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(health.router)                    # /health, /ready (no prefix — K8s probes)
    app.include_router(query.router, prefix=prefix)      # /api/v1/query
    app.include_router(ingest.router, prefix=prefix)     # /api/v1/ingest

    return app


app = create_app()
