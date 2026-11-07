"""Production FastAPI backend for the enterprise knowledge assistant RAG system.

Features:
- JWT Authentication and authorization
- Full LangGraph RAG pipeline integration with hybrid search & reranking
- Safety, prompt injection, and PII guardrails
- OpenTelemetry distributed tracing across all pipeline steps
- Prometheus metrics collection and /metrics exposition endpoint
- SQLite persistence for messages, feedback, sessions, and trace records
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import sys
import time
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

from auth import get_current_user, router as auth_router
from config import settings
from database import (
    get_csat_stats,
    get_recent_traces,
    get_stats,
    init_db,
    save_feedback,
    save_message,
    save_trace,
)
from models import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    GuardrailResult,
    HealthResponse,
    StatsResponse,
    TraceRecord,
)
from rag_pipeline import initialize_qdrant_collection, run_pipeline
from tracing import SpanAttr, create_rag_span, get_tracer, init_tracing

# Configure structured logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("knowledge_assistant.main")


# -------------------------------------------------------------------------
# Prometheus Metrics Definition
# -------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "knowledge_assistant_requests_total",
    "Total number of RAG chat requests processed",
    ["status"],
)

REQUEST_DURATION_MS = Histogram(
    "knowledge_assistant_request_duration_ms",
    "End-to-end request latency distribution in milliseconds",
    buckets=[100, 250, 500, 1000, 2000, 5000],
)

ERRORS_TOTAL = Counter(
    "knowledge_assistant_errors_total",
    "Total system errors classified by error type",
    ["error_type"],
)

ACTIVE_SESSIONS = Gauge(
    "knowledge_assistant_active_sessions",
    "Number of active user sessions in the past 24 hours",
)

GUARDRAIL_BLOCKS_TOTAL = Counter(
    "knowledge_assistant_guardrail_blocks_total",
    "Total queries intercepted by input or output guardrails",
    ["threat_type"],
)

LLM_FAILURES_TOTAL = Counter(
    "knowledge_assistant_llm_failures_total",
    "Total failed LLM calls during HyDE, reranking, or answer generation",
)

CSAT_POSITIVE = Gauge(
    "knowledge_assistant_csat_positive",
    "Total positive feedback responses received",
)

CSAT_TOTAL = Gauge(
    "knowledge_assistant_csat_total",
    "Total feedback responses received",
)

STAGE_DURATION_MS = Histogram(
    "knowledge_assistant_stage_duration_ms",
    "Execution latency of individual RAG pipeline stages in milliseconds",
    ["stage"],
    buckets=[25, 50, 100, 250, 500, 1000, 2500],
)


# -------------------------------------------------------------------------
# Application Lifespan
# -------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle hooks."""
    logger.info("Initializing Enterprise Knowledge Assistant API service (v%s)...", settings.APP_VERSION)

    # 1. Initialize database schema
    await init_db()

    # 3. Verify Qdrant connection and collection
    initialize_qdrant_collection()

    # 4. Prepopulate CSAT gauge metrics from persistent DB
    try:
        csat_data = await get_csat_stats()
        CSAT_POSITIVE.set(csat_data["positive"])
        CSAT_TOTAL.set(csat_data["total"])
    except Exception as exc:
        logger.warning("Could not initialize CSAT gauges: %s", exc)

    logger.info("Application startup completed successfully.")
    yield
    logger.info("Application shutting down...")


# -------------------------------------------------------------------------
# FastAPI App Initialization & Middleware
# -------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise Internal Knowledge Assistant RAG Backend Service",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Instrument before the application starts; Starlette does not allow middleware
# to be added from within the lifespan startup hook.
init_tracing(
    app_version=settings.APP_VERSION,
    service_name=settings.OTEL_SERVICE_NAME,
    otlp_endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
    fastapi_app=app,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus exposition endpoint
metrics_asgi = make_asgi_app()
app.mount("/metrics", metrics_asgi)

# Include Authentication Router
app.include_router(auth_router)


# -------------------------------------------------------------------------
# Error Handlers
# -------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format validation errors into consistent JSON schema."""
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    ERRORS_TOTAL.labels(error_type="ValidationError").inc()
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Request validation failed"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle standard HTTP exceptions with structured response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch unhandled server errors and log with stack trace."""
    logger.exception("Unhandled server exception on %s: %s", request.url.path, exc)
    ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the system administrator."},
    )


# -------------------------------------------------------------------------
# Application Endpoints
# -------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Service Health Check",
)
async def health_check() -> HealthResponse:
    """Return current service health and operational status."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc),
        ollama_status="ready",
        qdrant_status="ready",
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["RAG Assistant"],
    summary="Submit query to LangGraph RAG pipeline",
)
async def chat(
    chat_request: ChatRequest,
    current_user: str = Depends(get_current_user),
) -> ChatResponse:
    """Process an enterprise knowledge query through the complete RAG pipeline.

    - Authenticates user via JWT
    - Spans distributed trace across pipeline nodes
    - Executes guardrails, retrieval, reranking, and generation
    - Persists telemetry and conversation turn to SQLite
    - Emits Prometheus duration and error metrics
    """
    request_id = str(uuid4())
    start_time = time.perf_counter()
    tracer = get_tracer()

    with create_rag_span(
        tracer,
        "rag.request",
        **{
            SpanAttr.REQUEST_ID: request_id,
            SpanAttr.SESSION_ID: chat_request.session_id,
            SpanAttr.APP_VERSION: settings.APP_VERSION,
            SpanAttr.QUERY: chat_request.query,
            "user.username": current_user,
        },
    ) as span:
        try:
            # Execute LangGraph pipeline
            pipeline_result = await run_pipeline(
                query=chat_request.query,
                session_id=chat_request.session_id,
                request_id=request_id,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000
            pipeline_result["latency_ms"] = round(latency_ms, 2)

            guardrail: GuardrailResult = pipeline_result["guardrail"]
            status_label = "blocked" if guardrail.blocked else "success"

            # Prometheus metrics tracking
            REQUESTS_TOTAL.labels(status=status_label).inc()
            REQUEST_DURATION_MS.observe(latency_ms)

            if guardrail.blocked:
                threat = guardrail.threat_type or "general_block"
                GUARDRAIL_BLOCKS_TOTAL.labels(threat_type=threat).inc()

            # Record turn in database
            await save_message(
                message_id=pipeline_result["message_id"],
                session_id=chat_request.session_id,
                query=chat_request.query,
                response=pipeline_result["response"],
                sources=pipeline_result["sources"],
                confidence_score=pipeline_result["confidence_score"],
                latency_ms=pipeline_result["latency_ms"],
                guardrail_blocked=guardrail.blocked,
                guardrail_reason=guardrail.reason,
                username=current_user,
            )

            # Record high-level audit trace
            await save_trace(
                request_id=request_id,
                session_id=chat_request.session_id,
                query_preview=chat_request.query[:120],
                latency_ms=pipeline_result["latency_ms"],
                status=status_label,
                guardrail_blocked=guardrail.blocked,
            )

            span.set_attribute(SpanAttr.RESPONSE_STATUS, status_label)
            span.set_attribute("rag.latency_ms", pipeline_result["latency_ms"])

            return ChatResponse(
                message_id=pipeline_result["message_id"],
                response=pipeline_result["response"],
                sources=pipeline_result["sources"],
                confidence_score=pipeline_result["confidence_score"],
                guardrail=guardrail,
                latency_ms=pipeline_result["latency_ms"],
                session_id=chat_request.session_id,
                request_id=request_id,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - start_time) * 1000
            REQUESTS_TOTAL.labels(status="error").inc()
            ERRORS_TOTAL.labels(error_type=type(exc).__name__).inc()
            REQUEST_DURATION_MS.observe(latency_ms)

            await save_trace(
                request_id=request_id,
                session_id=chat_request.session_id,
                query_preview=chat_request.query[:120],
                latency_ms=latency_ms,
                status="error",
                guardrail_blocked=False,
            )

            span.set_attribute(SpanAttr.RESPONSE_STATUS, "error")
            span.record_exception(exc)
            logger.exception("Error executing chat request %s: %s", request_id, exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while evaluating your request: {str(exc)}",
            )


@app.post(
    "/feedback",
    tags=["Feedback"],
    summary="Record user thumbs up/down feedback",
)
async def submit_feedback(
    feedback_req: FeedbackRequest,
    current_user: str = Depends(get_current_user),
) -> Dict[str, str]:
    """Record user satisfaction feedback and update live Prometheus CSAT gauges."""
    await save_feedback(
        message_id=feedback_req.message_id,
        session_id=feedback_req.session_id,
        username=current_user,
        feedback_type=feedback_req.feedback.value,
    )

    # Refresh Prometheus gauges
    try:
        csat = await get_csat_stats()
        CSAT_POSITIVE.set(csat["positive"])
        CSAT_TOTAL.set(csat["total"])
    except Exception as exc:
        logger.warning("Could not refresh CSAT gauges: %s", exc)

    return {"status": "ok"}


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Observability"],
    summary="System and RAG Performance Statistics",
)
async def get_system_stats(
    current_user: str = Depends(get_current_user),
) -> StatsResponse:
    """Return aggregated platform metrics including CSAT, active sessions, and p95 latency."""
    stats_data = await get_stats()
    ACTIVE_SESSIONS.set(stats_data["active_sessions"])

    return StatsResponse(
        active_sessions=stats_data["active_sessions"],
        p95_latency_ms=stats_data["p95_latency_ms"],
        error_rate_percent=stats_data["error_rate_percent"],
        csat_percent=stats_data["csat_percent"],
        total_queries_today=stats_data["total_queries_today"],
    )


@app.get(
    "/traces/recent",
    response_model=List[TraceRecord],
    tags=["Observability"],
    summary="Recent execution traces",
)
async def get_traces(
    limit: int = 10,
    current_user: str = Depends(get_current_user),
) -> List[TraceRecord]:
    """Retrieve the most recent request execution traces."""
    recent_traces = await get_recent_traces(limit=min(limit, 100))
    return [TraceRecord(**t) for t in recent_traces]
