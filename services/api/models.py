"""Data schemas and Pydantic models for the enterprise knowledge assistant API.

Defines schemas for authentication, chat interactions, guardrails, feedback,
system metrics, and distributed tracing records.
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """User credentials for authentication."""

    username: str = Field(..., description="Username for login")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """Authentication response payload containing JWT access token."""

    access_token: str = Field(..., description="JWT bearer access token")
    token_type: str = Field(default="bearer", description="Token type")
    username: str = Field(..., description="Authenticated username")
    expires_in: int = Field(..., description="Token validity duration in seconds")


class ChatRequest(BaseModel):
    """Inbound chat query request."""

    query: str = Field(..., min_length=1, max_length=1000, description="User question or prompt")
    session_id: str = Field(..., description="Client conversation session identifier")


class SourceDocument(BaseModel):
    """Citation and provenance information for a retrieved document chunk."""

    doc_id: str = Field(..., description="Unique document chunk identifier")
    title: str = Field(..., description="Title of the source document")
    excerpt: str = Field(..., description="Relevant text excerpt from document")
    source_path: str = Field(..., description="File path or URL of source document")
    relevance_score: float = Field(..., description="Computed relevance score (0.0 to 1.0)")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable")
    doc_type: str = Field(default="document", description="Document type (pdf, markdown, policy, etc.)")


class GuardrailResult(BaseModel):
    """Evaluation result from safety and compliance guardrails."""

    blocked: bool = Field(..., description="Whether the request/response was blocked")
    reason: Optional[str] = Field(default=None, description="Reason for violation if blocked or flagged")
    threat_type: Optional[str] = Field(
        default=None,
        description="Category of threat detected (e.g. prompt_injection, off_topic, toxicity, data_exfiltration, pii_detected)",
    )


class ChatResponse(BaseModel):
    """Final answer returned to the client from the RAG pipeline."""

    message_id: str = Field(..., description="Unique ID for this response message")
    response: str = Field(..., description="Generated answer text")
    sources: List[SourceDocument] = Field(default_factory=list, description="Referenced source citations")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence score based on retrieval/reranking (0-1)"
    )
    guardrail: GuardrailResult = Field(..., description="Guardrail validation verdict")
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    session_id: str = Field(..., description="Session identifier")
    request_id: str = Field(..., description="Distributed tracing request identifier")


class FeedbackType(str, Enum):
    """Type of feedback given by user."""

    positive = "positive"
    negative = "negative"


class FeedbackRequest(BaseModel):
    """User feedback on a generated response."""

    message_id: str = Field(..., description="Message ID being evaluated")
    feedback: FeedbackType = Field(..., description="Positive (thumbs up) or negative (thumbs down)")
    session_id: str = Field(..., description="Session identifier")


class StatsResponse(BaseModel):
    """Operational statistics and system metrics."""

    active_sessions: int = Field(..., description="Currently active user sessions")
    p95_latency_ms: float = Field(..., description="95th percentile latency in milliseconds")
    error_rate_percent: float = Field(..., description="Percentage of requests resulting in errors")
    csat_percent: float = Field(..., description="Customer satisfaction percentage from feedback")
    total_queries_today: int = Field(..., description="Total queries processed today")


class TraceRecord(BaseModel):
    """Summary of a single request execution trace."""

    request_id: str = Field(..., description="Unique request tracing ID")
    timestamp: datetime = Field(..., description="Timestamp when request occurred")
    query_preview: str = Field(..., description="Truncated preview of the query")
    latency_ms: float = Field(..., description="Execution latency in milliseconds")
    status: str = Field(..., description="Status (success, blocked, error)")
    guardrail_blocked: bool = Field(..., description="Whether request was blocked by guardrails")
    session_id: str = Field(..., description="Session ID")


class HealthResponse(BaseModel):
    """Service health status response."""

    status: str = Field(default="healthy", description="Service health status")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(..., description="Current server UTC timestamp")
    ollama_status: Optional[str] = Field(default="unknown", description="Ollama connectivity status")
    qdrant_status: Optional[str] = Field(default="unknown", description="Qdrant connectivity status")
