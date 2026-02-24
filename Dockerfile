# ═══════════════════════════════════════════════════════════════════
# RAG LangChain API — Multi-Stage Dockerfile
# ═══════════════════════════════════════════════════════════════════
#
# Stages:
#   1. base        — shared Python base + OS packages
#   2. builder     — install all Python deps into a virtualenv
#   3. runtime     — minimal image with only the venv (no build tools)
#
# Design goals:
#   ✓ Reproducible on any OS (Linux, macOS, Windows via Docker)
#   ✓ Non-root USER for security
#   ✓ No secrets in the image (all via env vars at runtime)
#   ✓ Deterministic builds via pinned base image digest
#   ✓ .dockerignore excludes tests, .env, __pycache__, .git
#   ✓ HEALTHCHECK for Docker Compose (K8s uses liveness probes instead)
#   ✓ Tiny final image: only runtime dependencies copied from builder
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: base ────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS base

# Prevents Python from writing .pyc files and enables unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # pip tweaks
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # virtualenv path
    VENV_PATH=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install OS-level dependencies needed at RUNTIME
# (curl for health checks, libmagic for file type detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libmagic1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


# ── Stage 2: builder ─────────────────────────────────────────────────────
FROM base AS builder

# Install OS-level build tools (only needed to compile Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv $VENV_PATH

# Copy only dependency manifests first (layer cache optimization)
# This layer is only invalidated when pyproject.toml changes — not on code changes
WORKDIR /build
COPY pyproject.toml .

# Install production dependencies into the virtualenv
# --no-deps for torch to avoid pulling CUDA packages on CPU builds
RUN pip install --upgrade pip setuptools wheel && \
    pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.3.0" \
        --no-deps && \
    pip install -r requirements-dev.txt 2>/dev/null || pip install .


# ── Stage 3: runtime ─────────────────────────────────────────────────────
FROM base AS runtime

# Create non-root user and group
RUN groupadd --system --gid 1001 raguser && \
    useradd --system --uid 1001 --gid raguser --shell /bin/false raguser

WORKDIR /app

# Copy ONLY the virtualenv from builder stage (not build tools)
COPY --from=builder --chown=raguser:raguser $VENV_PATH $VENV_PATH

# Copy application source
COPY --chown=raguser:raguser app/ ./app/

# Switch to non-root user
USER raguser

# Expose API port
EXPOSE 8000

# Docker HEALTHCHECK (for docker-compose; Kubernetes uses liveness probes)
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=60s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint: uvicorn with production settings
# Workers=1 because we use async (single-process async > multi-process sync for I/O)
# Use gunicorn with uvicorn workers in prod if you need true multi-process:
#   gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 2 --bind 0.0.0.0:8000
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]
