"""
app/config.py
─────────────
Central configuration via Pydantic Settings.
All values are read from environment variables (with .env fallback).
Never hard-code secrets — use Kubernetes Secrets or .env locally.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────
    app_name: str = "rag-langchain"
    app_version: str = "1.0.0"
    environment: Literal["dev", "qa", "prod"] = "dev"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── API ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1                    # uvicorn workers (use 1 with async)
    api_timeout: int = 120                  # request timeout in seconds
    cors_origins: list[str] = ["*"]         # restrict in prod

    # ── Qdrant (vector database) ───────────────────────────────────────────
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"
    qdrant_grpc_port: int = 6334
    qdrant_prefer_grpc: bool = False        # enable in prod for performance

    # ── Embeddings (local, open-source) ───────────────────────────────────
    embed_model_name: str = "BAAI/bge-small-en-v1.5"
    embed_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embed_batch_size: int = 32
    embed_normalize: bool = True

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ──────────────────────────────────────────────────────────
    retrieval_top_k: int = 20              # how many chunks to fetch
    retrieval_final_k: int = 4            # how many to pass to LLM after rerank
    retrieval_bm25_weight: float = 0.4    # weight for BM25 in hybrid fusion
    retrieval_dense_weight: float = 0.6   # weight for dense in hybrid fusion

    # ── Reranker ──────────────────────────────────────────────────────────
    reranker_model_name: str = "BAAI/bge-reranker-base"
    reranker_enabled: bool = True

    # ── Ollama (local LLM) ────────────────────────────────────────────────
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "llama3"
    ollama_temperature: float = 0.0       # 0 = deterministic, better for RAG
    ollama_num_ctx: int = 4096            # context window size
    ollama_timeout: int = 90              # seconds

    # ── Redis (cache) ─────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    redis_cache_ttl: int = 3600           # 1 hour default TTL
    cache_enabled: bool = True

    # ── Langfuse (observability — open-source) ────────────────────────────
    langfuse_enabled: bool = True
    langfuse_host: str = "http://langfuse:3000"
    langfuse_public_key: str = ""         # set via K8s Secret
    langfuse_secret_key: str = ""         # set via K8s Secret
    langfuse_flush_interval: float = 0.5  # seconds

    # ── Ingestion ─────────────────────────────────────────────────────────
    max_upload_size_mb: int = 50
    allowed_file_types: list[str] = ["pdf", "txt", "md", "docx"]

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        return v.lower()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings singleton.
    Use FastAPI's Depends(get_settings) to inject into routes.
    Cache means .env is only parsed once at startup.
    """
    return Settings()
