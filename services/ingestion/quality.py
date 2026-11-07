"""
Layer 1: Data Quality checks.
Runs checks on each document and chunk, exposes Prometheus metrics.
"""

from typing import List, Dict, Any, Set, Optional
import os
import re
import time
import hashlib
import random
from datetime import datetime
from prometheus_client import Counter, Gauge, Histogram

# Prometheus Metrics definition
ingestion_docs_total = Counter(
    'knowledge_assistant_ingestion_docs_total',
    'Total documents ingested'
)
ingestion_failed_total = Counter(
    'knowledge_assistant_ingestion_failed_total',
    'Failed ingestions',
    ['error_type']
)
chunk_quality_score = Histogram(
    'knowledge_assistant_chunk_quality_score',
    'Chunk quality scores',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
doc_freshness_days = Gauge(
    'knowledge_assistant_doc_freshness_days',
    'Days since document was last modified'
)
doc_freshness_percent = Gauge(
    'knowledge_assistant_doc_freshness_percent',
    'Percent of docs within freshness threshold'
)
index_coverage_percent = Gauge(
    'knowledge_assistant_index_coverage_percent',
    'Percent of docs retrievable from index'
)
parse_errors_total = Counter(
    'knowledge_assistant_parse_errors_total',
    'Parse errors',
    ['error_type']
)


def check_chunk_quality(chunk: Any) -> Dict[str, Any]:
    """
    Returns {score: float, issues: list[str]}
    Checks:
    - min length (100 chars)
    - no garbled text
    - sentence completeness
    - not mostly whitespace
    - not mostly numbers/symbols
    """
    content = chunk.content if hasattr(chunk, "content") else (chunk.get("content") if isinstance(chunk, dict) else str(chunk))
    is_complete = getattr(chunk, "is_complete", True) if hasattr(chunk, "is_complete") else (chunk.get("is_complete", True) if isinstance(chunk, dict) else True)

    issues = []
    score = 1.0

    if not content or not content.strip():
        chunk_quality_score.observe(0.0)
        return {"score": 0.0, "issues": ["Chunk is empty"]}

    text_len = len(content.strip())

    # 1. Minimum length (100 chars)
    if text_len < 100:
        score -= 0.35
        issues.append(f"Chunk shorter than 100 chars ({text_len} chars)")

    # 2. Garbled text check
    garbled_chars = sum(
        1 for ch in content
        if ch == '\ufffd' or (ord(ch) < 32 and ch not in ('\t', '\n', '\r'))
    )
    garbled_ratio = garbled_chars / max(1, text_len)
    if garbled_ratio > 0.03:
        score -= min(0.5, garbled_ratio * 10)
        issues.append(f"Garbled or non-printable text ({garbled_ratio:.1%})")

    # 3. Sentence completeness
    if not is_complete:
        score -= 0.15
        issues.append("Incomplete sentence or truncation detected at end of chunk")

    # 4. Whitespace ratio
    whitespace_count = sum(1 for ch in content if ch.isspace())
    whitespace_ratio = whitespace_count / max(1, text_len)
    if whitespace_ratio > 0.35:
        score -= 0.20
        issues.append(f"Excessive whitespace ratio ({whitespace_ratio:.1%})")

    # 5. Letters ratio (alphabetic vs symbols/digits)
    alpha_count = sum(1 for ch in content if ch.isalpha())
    alpha_ratio = alpha_count / max(1, text_len)
    if alpha_ratio < 0.35:
        score -= 0.25
        issues.append(f"Low alphabetic content / mostly numbers or symbols ({alpha_ratio:.1%})")

    final_score = round(max(0.0, min(1.0, score)), 3)
    chunk_quality_score.observe(final_score)

    return {"score": final_score, "issues": issues}


def check_document_freshness(source_path: str, freshness_days_threshold: int = 365) -> Dict[str, Any]:
    """
    Returns {is_fresh: bool, days_old: float, modified_at: str}
    """
    try:
        if os.path.exists(source_path):
            mtime = os.path.getmtime(source_path)
            mod_dt = datetime.fromtimestamp(mtime)
            days_old = (datetime.now() - mod_dt).total_seconds() / 86400.0
            modified_at = mod_dt.isoformat()
        else:
            days_old = 0.0
            modified_at = datetime.now().isoformat()

        is_fresh = days_old <= freshness_days_threshold
        doc_freshness_days.set(round(days_old, 2))

        return {
            "is_fresh": is_fresh,
            "days_old": round(days_old, 2),
            "modified_at": modified_at
        }
    except Exception as e:
        return {
            "is_fresh": True,
            "days_old": 0.0,
            "modified_at": datetime.now().isoformat(),
            "error": str(e)
        }


def check_duplicate(chunk_content: str, existing_hashes: Set[str]) -> bool:
    """
    Returns True if this is a duplicate chunk.
    Uses MD5 hash of normalized content.
    """
    normalized = re.sub(r'\s+', ' ', chunk_content.strip().lower())
    content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
    if content_hash in existing_hashes:
        return True
    existing_hashes.add(content_hash)
    return False


async def check_index_coverage(qdrant_client: Any, collection_name: str, doc_ids: List[str]) -> float:
    """
    Spot-check: retrieve a random sample of doc_ids from Qdrant.
    Returns fraction that are retrievable.
    """
    if not doc_ids or qdrant_client is None:
        index_coverage_percent.set(100.0)
        return 1.0

    sample_size = min(len(doc_ids), 20)
    sampled_ids = random.sample(doc_ids, sample_size)
    retrieved_count = 0

    for doc_id in sampled_ids:
        try:
            # Query Qdrant with filter on payload doc_id
            search_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {"key": "doc_id", "match": {"value": doc_id}}
                    ]
                },
                limit=1
            )
            points, _ = search_result
            if points and len(points) > 0:
                retrieved_count += 1
        except Exception:
            # If Qdrant scroll fails or collection doesn't exist yet
            pass

    coverage = retrieved_count / sample_size if sample_size > 0 else 1.0
    index_coverage_percent.set(round(coverage * 100.0, 2))
    return coverage


async def run_quality_report(doc: Any, chunks: List[Any], qdrant_client: Any, collection_name: str) -> Dict[str, Any]:
    """
    Run all checks and return a quality report dict.
    Update Prometheus metrics.
    """
    source_path = getattr(doc, "source_path", "")
    parse_errors = getattr(doc, "parse_errors", [])
    doc_type = getattr(doc, "doc_type", "unknown")

    # Ingestion metrics
    if parse_errors:
        for err in parse_errors:
            error_key = "parse_exception" if "error" in err.lower() else "parse_warning"
            parse_errors_total.labels(error_type=error_key).inc()
            ingestion_failed_total.labels(error_type=error_key).inc()
    else:
        ingestion_docs_total.inc()

    # Freshness
    freshness = check_document_freshness(source_path)

    # Chunk quality checks
    chunk_scores = []
    chunk_issues = []
    existing_hashes: Set[str] = set()
    duplicate_count = 0

    for chunk in chunks:
        q_res = check_chunk_quality(chunk)
        chunk_scores.append(q_res["score"])
        if q_res["issues"]:
            chunk_issues.append({"chunk_id": getattr(chunk, "chunk_id", ""), "issues": q_res["issues"]})

        chunk_content = getattr(chunk, "content", "")
        if check_duplicate(chunk_content, existing_hashes):
            duplicate_count += 1

    avg_chunk_quality = (sum(chunk_scores) / len(chunk_scores)) if chunk_scores else 0.0

    report = {
        "source_path": source_path,
        "doc_type": doc_type,
        "parse_quality_score": getattr(doc, "parse_quality_score", 1.0),
        "parse_errors": parse_errors,
        "freshness": freshness,
        "chunk_count": len(chunks),
        "average_chunk_quality": round(avg_chunk_quality, 3),
        "duplicate_chunks_detected": duplicate_count,
        "chunk_issues": chunk_issues[:10],  # sample up to 10 issues
        "timestamp": datetime.now().isoformat()
    }

    return report
