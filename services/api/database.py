"""Asynchronous SQLite storage layer for messages, feedback, sessions, and traces.

Uses aiosqlite for non-blocking database queries, providing metrics persistence,
CSAT score calculation, latency percentile tracking, and trace history auditing.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any, Dict, List, Optional, Union

import aiosqlite
import numpy as np

from config import settings

logger = logging.getLogger("knowledge_assistant.database")


def _get_db_path() -> str:
    """Return configured SQLite database file path."""
    return settings.SQLITE_DB_PATH


async def init_db() -> None:
    """Initialize database tables, pragmas, and indexes if they do not exist."""
    db_path = _get_db_path()
    logger.info("Initializing SQLite database at: %s", db_path)

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA journal_mode = WAL;")
        await db.execute("PRAGMA synchronous = NORMAL;")
        await db.execute("PRAGMA foreign_keys = ON;")

        # Messages table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                sources TEXT,
                confidence_score REAL,
                latency_ms REAL,
                guardrail_blocked BOOLEAN,
                guardrail_reason TEXT,
                timestamp DATETIME NOT NULL
            );
            """
        )

        # Feedback table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                username TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            );
            """
        )

        # Sessions table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_active DATETIME NOT NULL,
                message_count INTEGER DEFAULT 0
            );
            """
        )

        # Traces table
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS traces (
                request_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                query_preview TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                status TEXT NOT NULL,
                guardrail_blocked BOOLEAN NOT NULL,
                timestamp DATETIME NOT NULL
            );
            """
        )

        # Indexes for fast querying
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_feedback_message ON feedback(message_id);")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_traces_timestamp ON traces(timestamp);")

        await db.commit()
    logger.info("Database schema verified and indexes prepared.")


async def save_message(
    message_id: str,
    session_id: str,
    query: str,
    response: str,
    sources: Union[List[dict], Any],
    confidence_score: float,
    latency_ms: float,
    guardrail_blocked: bool,
    guardrail_reason: Optional[str] = None,
    username: str = "anonymous",
) -> None:
    """Persist a conversation turn to the messages and sessions tables.

    Args:
        message_id: Unique message ID.
        session_id: Client conversation session ID.
        query: Inbound user prompt.
        response: Output answer text.
        sources: List of source documents or serialized JSON.
        confidence_score: Score between 0 and 1.
        latency_ms: Execution duration in milliseconds.
        guardrail_blocked: Whether request/response was blocked.
        guardrail_reason: Reason for guardrail intervention.
        username: Authenticated username.
    """
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    # Serialize sources if needed
    if isinstance(sources, list):
        serialized_sources = json.dumps([
            s.model_dump() if hasattr(s, "model_dump") else s for s in sources
        ])
    elif isinstance(sources, str):
        serialized_sources = sources
    else:
        serialized_sources = json.dumps([])

    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute(
            """
            INSERT INTO messages (
                message_id, session_id, query, response, sources,
                confidence_score, latency_ms, guardrail_blocked,
                guardrail_reason, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                message_id,
                session_id,
                query,
                response,
                serialized_sources,
                confidence_score,
                latency_ms,
                1 if guardrail_blocked else 0,
                guardrail_reason,
                now_iso,
            ),
        )

        # Upsert session record
        await db.execute(
            """
            INSERT INTO sessions (session_id, username, created_at, last_active, message_count)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(session_id) DO UPDATE SET
                last_active = excluded.last_active,
                message_count = sessions.message_count + 1;
            """,
            (session_id, username, now_iso, now_iso),
        )

        await db.commit()


async def save_feedback(
    message_id: str,
    session_id: str,
    username: str,
    feedback_type: str,
) -> None:
    """Record positive or negative feedback for a message.

    Args:
        message_id: Evaluated message ID.
        session_id: Associated session ID.
        username: Reviewer username.
        feedback_type: 'positive' or 'negative'.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute(
            """
            INSERT INTO feedback (message_id, session_id, username, feedback_type, timestamp)
            VALUES (?, ?, ?, ?, ?);
            """,
            (message_id, session_id, username, feedback_type, now_iso),
        )
        await db.commit()
    logger.info("Feedback '%s' saved for message '%s' by '%s'", feedback_type, message_id, username)


async def get_csat_stats() -> Dict[str, Any]:
    """Calculate Customer Satisfaction (CSAT) metrics from submitted feedback.

    Returns:
        Dictionary with keys 'positive', 'total', and 'percent'.
    """
    async with aiosqlite.connect(_get_db_path()) as db:
        async with db.execute(
            """
            SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type;
            """
        ) as cursor:
            rows = await cursor.fetchall()

    counts = {row[0]: row[1] for row in rows}
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    total = positive + negative
    percent = round((positive / total) * 100.0, 1) if total > 0 else 100.0

    return {
        "positive": positive,
        "negative": negative,
        "total": total,
        "percent": percent,
    }


async def save_trace(
    request_id: str,
    session_id: str,
    query_preview: str,
    latency_ms: float,
    status: str,
    guardrail_blocked: bool,
) -> None:
    """Persist high-level trace audit record for monitoring and debugging.

    Args:
        request_id: Distributed trace ID.
        session_id: Session ID.
        query_preview: Short preview snippet of query.
        latency_ms: Total latency in ms.
        status: Operation status (e.g. 'success', 'blocked', 'error').
        guardrail_blocked: Boolean indicating whether guardrails fired.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    truncated_query = query_preview[:120] if query_preview else ""

    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO traces (
                request_id, session_id, query_preview, latency_ms,
                status, guardrail_blocked, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                request_id,
                session_id,
                truncated_query,
                latency_ms,
                status,
                1 if guardrail_blocked else 0,
                now_iso,
            ),
        )
        await db.commit()


async def get_recent_traces(limit: int = 10) -> List[Dict[str, Any]]:
    """Retrieve recent trace records ordered newest first.

    Args:
        limit: Maximum number of trace records to retrieve.

    Returns:
        List of trace dictionaries formatted for TraceRecord schema.
    """
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT request_id, timestamp, query_preview, latency_ms, status,
                   guardrail_blocked, session_id
            FROM traces
            ORDER BY timestamp DESC
            LIMIT ?;
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append({
            "request_id": r["request_id"],
            "timestamp": datetime.fromisoformat(r["timestamp"]),
            "query_preview": r["query_preview"],
            "latency_ms": float(r["latency_ms"]),
            "status": r["status"],
            "guardrail_blocked": bool(r["guardrail_blocked"]),
            "session_id": r["session_id"],
        })
    return results


async def get_latency_percentiles() -> Dict[str, float]:
    """Calculate p50, p95, and p99 latency metrics across the last 1,000 queries.

    Returns:
        Dictionary with 'p50', 'p95', and 'p99' latencies in milliseconds.
    """
    async with aiosqlite.connect(_get_db_path()) as db:
        async with db.execute(
            """
            SELECT latency_ms FROM messages
            ORDER BY timestamp DESC
            LIMIT 1000;
            """
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    latencies = [row[0] for row in rows if row[0] is not None]
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    return {
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2),
    }


async def get_stats() -> Dict[str, Any]:
    """Aggregate high-level operational statistics for the system dashboard.

    Returns:
        Dictionary matching the StatsResponse model requirements.
    """
    async with aiosqlite.connect(_get_db_path()) as db:
        # Active sessions in the past 24 hours
        async with db.execute(
            """
            SELECT COUNT(*) FROM sessions
            WHERE datetime(last_active) >= datetime('now', '-24 hours');
            """
        ) as cursor:
            active_sessions_row = await cursor.fetchone()
            active_sessions = active_sessions_row[0] if active_sessions_row else 0

        # Total queries today
        async with db.execute(
            """
            SELECT COUNT(*) FROM messages
            WHERE date(timestamp) = date('now');
            """
        ) as cursor:
            queries_today_row = await cursor.fetchone()
            total_queries_today = queries_today_row[0] if queries_today_row else 0

        # Total messages and blocked/error messages
        async with db.execute(
            """
            SELECT
                COUNT(*),
                SUM(CASE WHEN guardrail_blocked = 1 THEN 1 ELSE 0 END)
            FROM messages;
            """
        ) as cursor:
            tot_row = await cursor.fetchone()
            total_messages = tot_row[0] if tot_row else 0
            blocked_messages = tot_row[1] if (tot_row and tot_row[1]) else 0

    error_rate = round((blocked_messages / total_messages) * 100.0, 2) if total_messages > 0 else 0.0
    latencies = await get_latency_percentiles()
    csat = await get_csat_stats()

    return {
        "active_sessions": active_sessions,
        "p95_latency_ms": latencies["p95"],
        "error_rate_percent": error_rate,
        "csat_percent": csat["percent"],
        "total_queries_today": total_queries_today,
    }
