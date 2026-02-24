"""
app/observability/langfuse.py
──────────────────────────────
Langfuse observability integration (100% open-source).
Provides:
  - LangfuseCallbackHandler for automatic LangChain tracing
  - Helpers to create spans, scores, and flush traces
  - A no-op fallback when Langfuse is disabled (dev/test)

Self-hosted Langfuse: https://langfuse.com/docs/deployment/self-host
Docker image: ghcr.io/langfuse/langfuse:latest
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler

logger = logging.getLogger(__name__)


class NoOpTrace:
    """Returned when Langfuse is disabled — all methods are no-ops."""

    trace_id: str = "disabled"

    def update(self, **_: Any) -> None: ...
    def score(self, **_: Any) -> None: ...
    def span(self, **_: Any) -> "NoOpTrace": return self
    def end(self, **_: Any) -> None: ...
    def __enter__(self) -> "NoOpTrace": return self
    def __exit__(self, *_: Any) -> None: ...


class ObservabilityClient:
    """
    Thin wrapper around the Langfuse SDK.
    Instantiated once at startup and injected via FastAPI dependencies.

    Usage:
        # In a route handler
        trace = obs_client.trace(name="rag-query", session_id=str(session_id))
        handler = obs_client.callback_handler(trace_id=trace.id)
        chain.invoke(input, config={"callbacks": [handler]})
        obs_client.flush()
    """

    def __init__(
        self,
        enabled: bool,
        public_key: str,
        secret_key: str,
        host: str,
        flush_interval: float = 0.5,
    ) -> None:
        self._enabled = enabled
        self._client: Any = None

        if enabled and public_key and secret_key:
            try:
                from langfuse import Langfuse  # type: ignore[import]

                self._client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                    flush_interval=flush_interval,
                )
                logger.info("Langfuse observability enabled", extra={"host": host})
            except Exception as exc:
                logger.warning(
                    "Langfuse init failed — observability disabled",
                    extra={"error": str(exc)},
                )
                self._enabled = False
        else:
            if enabled:
                logger.warning(
                    "Langfuse keys not configured — running without observability"
                )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    def trace(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """Create a new Langfuse trace. Returns NoOpTrace if disabled."""
        if not self.enabled:
            return NoOpTrace()
        try:
            return self._client.trace(
                name=name,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
                tags=tags or [],
            )
        except Exception as exc:
            logger.warning("Failed to create Langfuse trace", extra={"error": str(exc)})
            return NoOpTrace()

    def callback_handler(
        self,
        trace_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """
        Return a LangChain callback handler that auto-traces all chain steps.
        Pass this to chain.invoke(..., config={"callbacks": [handler]}).
        Returns a no-op handler if Langfuse is disabled.
        """
        if not self.enabled:
            return _NoOpCallbackHandler()
        try:
            from langfuse.callback import CallbackHandler  # type: ignore[import]

            return CallbackHandler(
                public_key=self._client.public_key,
                secret_key=self._client.secret_key,
                host=self._client.host,
                trace_id=trace_id,
                session_id=session_id,
                user_id=user_id,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.warning("Failed to create Langfuse callback", extra={"error": str(exc)})
            return _NoOpCallbackHandler()

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        comment: str | None = None,
    ) -> None:
        """Attach a quality score to a trace (e.g. faithfulness from RAGAS eval)."""
        if not self.enabled:
            return
        try:
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:
            logger.warning("Failed to record Langfuse score", extra={"error": str(exc)})

    def flush(self) -> None:
        """Flush all pending events to Langfuse. Call after each request."""
        if not self.enabled:
            return
        try:
            self._client.flush()
        except Exception as exc:
            logger.debug("Langfuse flush error", extra={"error": str(exc)})

    def shutdown(self) -> None:
        """Flush and shutdown. Call on app shutdown."""
        if not self.enabled:
            return
        try:
            self._client.flush()
            logger.info("Langfuse client shut down cleanly")
        except Exception as exc:
            logger.warning("Langfuse shutdown error", extra={"error": str(exc)})

    @contextmanager
    def trace_context(
        self,
        name: str,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Generator[tuple[Any, Any], None, None]:
        """
        Context manager that creates a trace + callback handler together.

        Usage:
            async with obs_client.trace_context("rag-query", session_id=sid) as (trace, handler):
                result = await chain.ainvoke(input, config={"callbacks": [handler]})
                obs_client.flush()
        """
        trace = self.trace(name=name, session_id=session_id, metadata=metadata)
        handler = self.callback_handler(
            trace_id=getattr(trace, "id", str(uuid.uuid4())),
            session_id=session_id,
        )
        try:
            yield trace, handler
        except Exception as exc:
            if hasattr(trace, "update"):
                trace.update(status="error", status_message=str(exc))
            raise
        finally:
            self.flush()


class _NoOpCallbackHandler:
    """Minimal no-op that satisfies the LangChain callback interface."""

    def on_chain_start(self, *_: Any, **__: Any) -> None: ...
    def on_chain_end(self, *_: Any, **__: Any) -> None: ...
    def on_chain_error(self, *_: Any, **__: Any) -> None: ...
    def on_llm_start(self, *_: Any, **__: Any) -> None: ...
    def on_llm_end(self, *_: Any, **__: Any) -> None: ...
    def on_llm_error(self, *_: Any, **__: Any) -> None: ...
    def on_retriever_start(self, *_: Any, **__: Any) -> None: ...
    def on_retriever_end(self, *_: Any, **__: Any) -> None: ...
    def on_retriever_error(self, *_: Any, **__: Any) -> None: ...
