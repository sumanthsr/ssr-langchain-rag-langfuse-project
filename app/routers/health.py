"""
app/routers/health.py
──────────────────────
Health and readiness endpoints used by Kubernetes probes.

  GET /health  — liveness probe (is the process alive?)
  GET /ready   — readiness probe (is the app ready to serve traffic?)

Kubernetes will:
  - Restart the pod if /health returns non-200 (liveness)
  - Stop sending traffic if /ready returns non-200 (readiness)
  - Remove pod from Service endpoints until /ready returns 200 again

These are also exposed to Prometheus via the /metrics endpoint (if enabled).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Response, status

from app.dependencies import ObsDep, RedisDep, SettingsDep, get_vector_store
from app.models.schemas import DependencyStatus, HealthResponse, ReadinessResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

# Record startup time for uptime calculation
_startup_time = time.time()


async def _check_qdrant(vector_store: Any) -> DependencyStatus:
    """Ping Qdrant by listing collections."""
    t0 = time.perf_counter()
    try:
        vector_store.client.get_collections()
        latency = (time.perf_counter() - t0) * 1000
        return DependencyStatus(name="qdrant", status="ok", latency_ms=round(latency, 1))
    except Exception as exc:
        return DependencyStatus(name="qdrant", status="down", detail=str(exc))


async def _check_ollama(ollama_url: str) -> DependencyStatus:
    """Ping Ollama /api/tags endpoint."""
    import httpx

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return DependencyStatus(name="ollama", status="ok", latency_ms=round(latency, 1))
        return DependencyStatus(
            name="ollama", status="degraded", detail=f"HTTP {resp.status_code}"
        )
    except Exception as exc:
        return DependencyStatus(name="ollama", status="down", detail=str(exc))


async def _check_redis(redis: Any) -> DependencyStatus:
    """Ping Redis."""
    if redis is None:
        return DependencyStatus(name="redis", status="ok", detail="disabled")
    t0 = time.perf_counter()
    try:
        redis.ping()
        latency = (time.perf_counter() - t0) * 1000
        return DependencyStatus(name="redis", status="ok", latency_ms=round(latency, 1))
    except Exception as exc:
        return DependencyStatus(name="redis", status="degraded", detail=str(exc))


async def _check_langfuse(obs: Any) -> DependencyStatus:
    """Check Langfuse connectivity."""
    if not obs.enabled:
        return DependencyStatus(name="langfuse", status="ok", detail="disabled")
    t0 = time.perf_counter()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{obs._client.host}/api/public/health")  # noqa: SLF001
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return DependencyStatus(name="langfuse", status="ok", latency_ms=round(latency, 1))
        return DependencyStatus(name="langfuse", status="degraded", detail=f"HTTP {resp.status_code}")
    except Exception as exc:
        return DependencyStatus(name="langfuse", status="degraded", detail=str(exc))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe — is this pod alive?",
)
async def health(
    settings: SettingsDep,
    redis: RedisDep,
    obs: ObsDep,
    response: Response,
) -> HealthResponse:
    """
    Kubernetes liveness probe.
    Returns 200 if the process is alive, 503 if critical dependencies are down.
    Keep this fast (<500ms) — K8s probes have tight timeouts.
    """
    vector_store = get_vector_store()

    checks = await _gather_checks(vector_store, settings, redis, obs)
    overall_status = _compute_status(checks)

    if overall_status == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(time.time() - _startup_time, 1),
        dependencies=checks,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe — is this pod ready to serve traffic?",
)
async def ready(
    settings: SettingsDep,
    redis: RedisDep,
    obs: ObsDep,
    response: Response,
) -> ReadinessResponse:
    """
    Kubernetes readiness probe.
    Returns 200 only when ALL critical dependencies (Qdrant, Ollama) are reachable.
    K8s will stop sending traffic to this pod if this returns 503.
    """
    vector_store = get_vector_store()
    checks = await _gather_checks(vector_store, settings, redis, obs)

    check_map = {c.name: c.status == "ok" for c in checks}
    is_ready = check_map.get("qdrant", False) and check_map.get("ollama", False)

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("Readiness check failed", extra={"checks": check_map})

    return ReadinessResponse(ready=is_ready, checks=check_map)


async def _gather_checks(
    vector_store: Any,
    settings: SettingsDep,
    redis: Any,
    obs: Any,
) -> list[DependencyStatus]:
    import asyncio

    results = await asyncio.gather(
        _check_qdrant(vector_store),
        _check_ollama(settings.ollama_base_url),
        _check_redis(redis),
        _check_langfuse(obs),
        return_exceptions=True,
    )
    return [r if isinstance(r, DependencyStatus) else DependencyStatus(name="unknown", status="down", detail=str(r)) for r in results]


def _compute_status(checks: list[DependencyStatus]) -> str:
    statuses = {c.status for c in checks}
    if "down" in statuses:
        return "down"
    if "degraded" in statuses:
        return "degraded"
    return "ok"
