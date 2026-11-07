"""OpenTelemetry distributed tracing setup and instrumentation for RAG pipeline.

Initializes the TracerProvider with OTLP gRPC export, provides semantic span attribute
constants, and offers context managers for granular span timing and annotation.
"""

from contextlib import contextmanager
import logging
from typing import Any, Dict, Generator, List, Optional, Union

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Status, StatusCode, Tracer

logger = logging.getLogger("knowledge_assistant.tracing")

_TRACER_NAME = "knowledge_assistant.rag"


class SpanAttr:
    """Semantic convention constants for RAG pipeline distributed spans."""

    REQUEST_ID = "rag.request_id"
    SESSION_ID = "rag.session_id"
    APP_VERSION = "app.version"
    QUERY = "rag.query"
    REWRITTEN_QUERY = "rag.rewritten_query"
    RETRIEVAL_DURATION_MS = "rag.retrieval_duration_ms"
    NUM_DOCS_RETRIEVED = "rag.num_docs_retrieved"
    DOC_IDS = "rag.doc_ids"
    RELEVANCE_SCORES = "rag.relevance_scores"
    PROMPT_VERSION = "rag.prompt_version"
    CONTEXT_SIZE_TOKENS = "rag.context_size_tokens"
    LLM_MODEL = "llm.model"
    LLM_DURATION_MS = "llm.duration_ms"
    LLM_PROMPT_TOKENS = "llm.prompt_tokens"
    LLM_COMPLETION_TOKENS = "llm.completion_tokens"
    LLM_STATUS = "llm.status"
    LLM_FINISH_REASON = "llm.finish_reason"
    TOOL_NAME = "tool.name"
    TOOL_DURATION_MS = "tool.duration_ms"
    TOOL_STATUS = "tool.status"
    OUTPUT_TOKEN_COUNT = "generation.output_tokens"
    RESPONSE_STATUS = "response.status"
    EVAL_SCORE = "response.eval_score"
    FEEDBACK = "response.feedback"
    GUARDRAIL_BLOCKED = "guardrail.blocked"
    GUARDRAIL_THREAT_TYPE = "guardrail.threat_type"
    PII_DETECTED = "guardrail.pii_detected"


_initialized = False


def init_tracing(
    app_version: str,
    service_name: str,
    otlp_endpoint: str,
    fastapi_app: Any = None,
) -> Tracer:
    """Initialize OpenTelemetry with OTLP gRPC exporter. Call once at startup.

    Configures TracerProvider, Resource metadata, BatchSpanProcessor, and optionally
    auto-instruments FastAPI routes.

    Args:
        app_version: Current application version string.
        service_name: Service name reported in traces.
        otlp_endpoint: OTLP gRPC collector endpoint (e.g. http://otel-collector:4317).
        fastapi_app: Optional FastAPI instance to instrument automatically.

    Returns:
        Configured OpenTelemetry Tracer instance.
    """
    global _initialized
    if _initialized:
        logger.info("OpenTelemetry already initialized. Returning existing tracer.")
        return get_tracer()

    logger.info(
        "Initializing OpenTelemetry for service '%s' (v%s) targeting '%s'",
        service_name,
        app_version,
        otlp_endpoint,
    )

    resource = Resource.create(
        attributes={
            "service.name": service_name,
            "service.version": app_version,
            "deployment.environment": "production",
        }
    )

    provider = TracerProvider(resource=resource)

    try:
        # Strip scheme if present for gRPC endpoint connection string if needed,
        # but OTLPSpanExporter handles standard uri endpoints.
        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=True,
        )
        span_processor = BatchSpanProcessor(
            exporter,
            max_queue_size=2048,
            schedule_delay_millis=500,
            max_export_batch_size=512,
        )
        provider.add_span_processor(span_processor)
        logger.info("BatchSpanProcessor and OTLPSpanExporter registered successfully.")
    except Exception as exc:
        logger.warning(
            "Failed to connect OTLPSpanExporter to endpoint %s: %s. Traces will be collected locally.",
            otlp_endpoint,
            exc,
        )

    trace.set_tracer_provider(provider)

    if fastapi_app is not None:
        try:
            FastAPIInstrumentor.instrument_app(
                fastapi_app,
                tracer_provider=provider,
                excluded_urls="health,metrics",
            )
            logger.info("FastAPI auto-instrumentation attached to application.")
        except Exception as exc:
            logger.warning("Could not auto-instrument FastAPI: %s", exc)

    _initialized = True
    return get_tracer()


def get_tracer() -> Tracer:
    """Get the global application tracer for manual instrumentation."""
    return trace.get_tracer(_TRACER_NAME)


def _set_span_attribute(span: trace.Span, key: str, val: Any) -> None:
    """Safely cast and assign an attribute value to an OpenTelemetry span."""
    if val is None:
        return
    if isinstance(val, (bool, str, int, float)):
        span.set_attribute(key, val)
    elif isinstance(val, (list, tuple)):
        # If homogeneous primitive list, OTEL supports it
        if all(isinstance(x, str) for x in val):
            span.set_attribute(key, [str(x) for x in val])
        elif all(isinstance(x, (int, float)) for x in val):
            span.set_attribute(key, [float(x) for x in val])
        elif all(isinstance(x, bool) for x in val):
            span.set_attribute(key, [bool(x) for x in val])
        else:
            span.set_attribute(key, [str(x) for x in val])
    elif isinstance(val, dict):
        span.set_attribute(key, str(val))
    else:
        span.set_attribute(key, str(val))


@contextmanager
def create_rag_span(
    tracer: Optional[Tracer] = None,
    span_name: str = "rag.operation",
    parent_context: Any = None,
    **attributes: Any,
) -> Generator[trace.Span, None, None]:
    """Context manager that creates a child span with given attributes and exception tracking.

    Args:
        tracer: Optional Tracer instance, defaults to global tracer.
        span_name: Name of the span operation.
        parent_context: Optional parent trace context.
        **attributes: Key-value attributes to associate with the span.

    Yields:
        Active OpenTelemetry Span.
    """
    active_tracer = tracer if tracer is not None else get_tracer()
    with active_tracer.start_as_current_span(span_name, context=parent_context) as span:
        for key, val in attributes.items():
            _set_span_attribute(span, key, val)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
