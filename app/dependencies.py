"""
app/dependencies.py
────────────────────
FastAPI dependency injection container.
All expensive objects (embeddings model, vector store, Langfuse client)
are initialised ONCE at startup and reused across requests.

Pattern: module-level singletons initialised in app lifespan,
then exposed via Depends() functions.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import Depends

from app.config import Settings, get_settings
from app.observability.langfuse import ObservabilityClient
from app.services.chain import RAGChain
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

# ── Singletons (populated during app lifespan startup) ─────────────────────
_vector_store: Any = None
_embeddings: Any = None
_obs_client: ObservabilityClient | None = None
_ingestion_service: IngestionService | None = None
_retrieval_service: RetrievalService | None = None
_rag_chain: RAGChain | None = None
_redis_client: Any = None


def init_embeddings(settings: Settings) -> Any:
    """Load HuggingFace embedding model (cached in process memory)."""
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

    logger.info("Loading embedding model", extra={"model": settings.embed_model_name})
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model_name,
        model_kwargs={"device": settings.embed_device},
        encode_kwargs={
            "normalize_embeddings": settings.embed_normalize,
            "batch_size": settings.embed_batch_size,
        },
    )


def init_vector_store(settings: Settings, embeddings: Any) -> Any:
    """Connect to Qdrant and ensure collection exists."""
    from langchain_qdrant import QdrantVectorStore  # type: ignore
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.models import Distance, VectorParams  # type: ignore

    logger.info("Connecting to Qdrant", extra={"url": settings.qdrant_url})
    client = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        grpc_port=settings.qdrant_grpc_port,
        prefer_grpc=settings.qdrant_prefer_grpc,
        timeout=10,
    )

    # Create collection if it doesn't exist
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        # Determine vector size from a test embed
        test_vec = embeddings.embed_query("test")
        vector_size = len(test_vec)
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(
            "Created Qdrant collection",
            extra={"collection": settings.qdrant_collection, "dims": vector_size},
        )

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def init_redis(settings: Settings) -> Any | None:
    """Connect to Redis for response caching. Returns None if disabled."""
    if not settings.cache_enabled:
        return None
    try:
        import redis  # type: ignore

        client = redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=3)
        client.ping()
        logger.info("Redis cache connected", extra={"url": settings.redis_url})
        return client
    except Exception as exc:
        logger.warning("Redis unavailable — caching disabled", extra={"error": str(exc)})
        return None


def init_all(settings: Settings) -> None:
    """
    Called once during FastAPI lifespan startup.
    Initialises all singletons in the correct dependency order.
    """
    global _embeddings, _vector_store, _obs_client
    global _ingestion_service, _retrieval_service, _rag_chain, _redis_client

    _embeddings = init_embeddings(settings)
    _vector_store = init_vector_store(settings, _embeddings)
    _redis_client = init_redis(settings)

    _obs_client = ObservabilityClient(
        enabled=settings.langfuse_enabled,
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
        flush_interval=settings.langfuse_flush_interval,
    )

    _ingestion_service = IngestionService(settings=settings, vector_store=_vector_store)
    _retrieval_service = RetrievalService(settings=settings, vector_store=_vector_store)
    _rag_chain = RAGChain(settings=settings)

    logger.info("All services initialised successfully")


def shutdown_all() -> None:
    """Called during FastAPI lifespan shutdown."""
    if _obs_client:
        _obs_client.shutdown()
    if _redis_client:
        _redis_client.close()
    logger.info("All services shut down")


# ── FastAPI Depends functions ───────────────────────────────────────────────

def get_vector_store() -> Any:
    assert _vector_store is not None, "Vector store not initialised"
    return _vector_store


def get_obs_client() -> ObservabilityClient:
    assert _obs_client is not None, "ObservabilityClient not initialised"
    return _obs_client


def get_ingestion_service() -> IngestionService:
    assert _ingestion_service is not None, "IngestionService not initialised"
    return _ingestion_service


def get_retrieval_service() -> RetrievalService:
    assert _retrieval_service is not None, "RetrievalService not initialised"
    return _retrieval_service


def get_rag_chain() -> RAGChain:
    assert _rag_chain is not None, "RAGChain not initialised"
    return _rag_chain


def get_redis() -> Any | None:
    return _redis_client


# ── Annotated type aliases for cleaner route signatures ────────────────────
SettingsDep = Annotated[Settings, Depends(get_settings)]
ObsDep = Annotated[ObservabilityClient, Depends(get_obs_client)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion_service)]
RetrievalDep = Annotated[RetrievalService, Depends(get_retrieval_service)]
ChainDep = Annotated[RAGChain, Depends(get_rag_chain)]
RedisDep = Annotated[Any | None, Depends(get_redis)]
