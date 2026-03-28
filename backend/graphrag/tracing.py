"""LangFuse-based tracing for GraphRAG pipeline operations."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Generator

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional fallback
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

if TYPE_CHECKING:
    from langfuse import Langfuse as LangfuseClient
else:
    LangfuseClient = Any

try:
    from langfuse import Langfuse
except ImportError:
    Langfuse = None


class TracingManager:
    """Manages LangFuse tracing for GraphRAG operations."""

    _instance: TracingManager | None = None

    def __new__(cls) -> TracingManager:
        """Singleton pattern for TracingManager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._initialized = True
        self.enabled = False
        self.langfuse: LangfuseClient | None = None
        self.current_trace: Any = None

        # Initialize LangFuse if credentials provided
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv(
            "LANGFUSE_HOST",
            os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )

        if public_key and secret_key and Langfuse is not None:
            try:
                self.langfuse = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host,
                )
                self.enabled = True
                print(f"✓ LangFuse tracing enabled (host={host})")
            except Exception as e:
                print(f"⚠ LangFuse initialization failed: {e}")
        elif not (public_key and secret_key):
            print(
                "ℹ LangFuse tracing disabled. Set LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY to enable."
            )

    @contextmanager
    def trace(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """
        Context manager for tracing operations.

        Args:
            name: Name of the operation
            metadata: Additional metadata to attach
            input_data: Input data to log

        Yields:
            Trace context with output tracking
        """
        if not self.enabled or self.langfuse is None:
            yield {"output": None, "metadata": {}}
            return

        start_time = time.time()
        trace = self.langfuse.trace(
            name=name,
            input=input_data,
            metadata=metadata,
        )
        trace_context = {
            "output": None,
            "metadata": metadata or {},
            "error": None,
        }

        try:
            if input_data:
                trace.event(name="input", input=input_data)

            yield trace_context

            # Record successful completion
            duration_ms = int((time.time() - start_time) * 1000)
            trace.update(
                input=input_data,
                output=trace_context.get("output"),
                metadata={
                    "duration_ms": duration_ms,
                    **(metadata or {}),
                },
            )

        except Exception as e:
            trace_context["error"] = str(e)
            trace.update(
                input=input_data,
                output={"error": str(e)},
                metadata={
                    "error": str(e),
                    **(metadata or {}),
                },
            )
            raise

    @contextmanager
    def span(
        self,
        name: str,
        parent_trace: Any = None,
        metadata: dict[str, Any] | None = None,
        input_data: Any = None,
    ) -> Generator[Any, None, None]:
        """
        Context manager for tracing nested spans.

        Args:
            name: Name of the span
            parent_trace: Parent trace object
            metadata: Additional metadata
            input_data: Input data to log

        Yields:
            Span object for output tracking
        """
        if not self.enabled or self.langfuse is None:
            yield None
            return

        span = parent_trace.span(name=name) if parent_trace else None
        if not span and self.langfuse:
            span = self.langfuse.trace(name=name, input=input_data, metadata=metadata)

        if span and input_data:
            span.event(name="input", input=input_data)

        try:
            yield span
            if span and hasattr(span, "end"):
                span.end()
            elif span:
                span.update(
                    output={"status": "completed"},
                    metadata=metadata,
                )
        except Exception as e:
            if span:
                if hasattr(span, "end"):
                    span.end(
                        output={"error": str(e)},
                        status_message=str(e),
                        metadata=metadata,
                    )
                else:
                    span.update(
                        output={"error": str(e)},
                        metadata={
                            "error": str(e),
                            **(metadata or {}),
                        },
                    )
            raise

    def log_llm_call(
        self,
        name: str,
        model: str,
        prompt: str | list[dict[str, str]],
        response: str,
        tokens_used: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an LLM API call.

        Args:
            name: Operation name
            model: Model identifier
            prompt: Input prompt or messages
            response: Model response
            tokens_used: Token usage stats (input, output, total)
            metadata: Additional metadata
        """
        if not self.enabled or self.langfuse is None:
            return

        self.langfuse.generation(
            name=name,
            model=model,
            input=prompt,
            output=response,
            usage={
                "input": tokens_used.get("input", 0) if tokens_used else None,
                "output": tokens_used.get("output", 0) if tokens_used else None,
            },
            metadata={
                "operation": name,
                **(metadata or {}),
            },
        )

    def log_retrieval(
        self,
        query: str,
        results: list[dict[str, Any]],
        result_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a retrieval operation.

        Args:
            query: Search query
            results: Retrieved results
            result_count: Number of results
            metadata: Additional metadata
        """
        if not self.enabled or self.langfuse is None:
            return

        self.langfuse.event(
            name="retrieval",
            input={"query": query},
            output={
                "result_count": result_count,
                "results": results[:3],  # Log first 3 results
            },
            metadata={
                "query_type": "vector_search",
                "result_count": result_count,
                **(metadata or {}),
            },
        )

    def log_entity_extraction(
        self,
        paper_id: str,
        entity_count: int,
        entity_types: dict[str, int],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log entity extraction results.

        Args:
            paper_id: Paper being processed
            entity_count: Total entities extracted
            entity_types: Breakdown by entity type
            metadata: Additional metadata
        """
        if not self.enabled or self.langfuse is None:
            return

        self.langfuse.event(
            name="entity_extraction",
            input={"paper_id": paper_id},
            output={
                "entity_count": entity_count,
                "entity_types": entity_types,
            },
            metadata={
                "paper_id": paper_id,
                **entity_types,
                **(metadata or {}),
            },
        )

    def log_edge_creation(
        self,
        edge_count: int,
        edge_types: dict[str, int],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Log edge creation results.

        Args:
            edge_count: Total edges created
            edge_types: Breakdown by edge type
            metadata: Additional metadata
        """
        if not self.enabled or self.langfuse is None:
            return

        self.langfuse.event(
            name="edge_creation",
            output={
                "edge_count": edge_count,
                "edge_types": edge_types,
            },
            metadata={
                "edge_count": edge_count,
                **edge_types,
                **(metadata or {}),
            },
        )

    def flush(self) -> None:
        """Flush all pending traces to LangFuse."""
        if self.enabled and self.langfuse:
            self.langfuse.flush()


# Global singleton manager
_manager: TracingManager | None = None


def get_tracing_manager() -> TracingManager:
    """Get or create the global tracing manager."""
    global _manager
    if _manager is None:
        _manager = TracingManager()
    return _manager
