"""
Custom evaluation metrics for layers not covered by RAGAS.
Covers:
- Layer 2: Retrieval metrics (Precision@k, Recall@k, MRR, NDCG@k)
- Layer 5: Business metrics (CSAT, Task Completion, Escalation Rate, Cost per Task)
"""

from typing import List, Dict, Any, Union
import numpy as np
import re

ESCALATION_PHRASES = [
    "contact hr",
    "contact it",
    "reach out to",
    "please email",
    "submit a ticket",
    "open a ticket",
    "escalate to",
    "contact the helpdesk",
]


# ============================================================================
# Layer 2: Retrieval Metrics
# ============================================================================

def precision_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    """
    Fraction of top-k retrieved docs that are relevant.
    Precision@k = (Number of relevant docs in top-k) / k
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    num_relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(float(num_relevant_retrieved / k), 4)


def recall_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    """
    Fraction of relevant docs that appear in top-k retrieved.
    Recall@k = (Number of relevant docs in top-k) / (Total relevant docs)
    """
    if not relevant_ids:
        return 0.0
    if k <= 0 or not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    num_relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return round(float(num_relevant_retrieved / len(relevant_set)), 4)


def mean_reciprocal_rank(retrieved_ids_list: List[List[Any]], relevant_ids_list: List[List[Any]]) -> float:
    """
    MRR over a list of queries.
    MRR = (1 / |Q|) * sum(1 / rank_i)
    """
    if not retrieved_ids_list or not relevant_ids_list:
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        relevant_set = set(relevant)
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return round(float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0, 4)


def ndcg_at_k(retrieved_ids: List[Any], relevant_ids: List[Any], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k (NDCG@k).
    Binary relevance: 1 if relevant else 0.
    """
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)

    # Calculate DCG@k
    dcg = 0.0
    for idx, doc_id in enumerate(top_k):
        if doc_id in relevant_set:
            # log2(idx + 2) because idx is 0-indexed, so rank = idx + 1
            dcg += 1.0 / np.log2(idx + 2)

    # Calculate Ideal DCG@k (all relevant items ranked at the top)
    ideal_hits = min(k, len(relevant_set))
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return round(float(dcg / idcg), 4)


# ============================================================================
# Layer 5: Business Metrics (computed from DB/API records)
# ============================================================================

def compute_csat(feedback_records: List[Dict[str, Any]]) -> Dict[str, Union[int, float]]:
    """
    Computes Customer Satisfaction (CSAT) from feedback records.
    Returns {positive: int, total: int, percent: float}
    """
    if not feedback_records:
        return {"positive": 0, "total": 0, "percent": 0.0}

    positive_count = 0
    total = len(feedback_records)

    for item in feedback_records:
        # Check various feedback schemas (thumbs, score 1-5, binary rating)
        if item.get("is_positive") is True:
            positive_count += 1
        elif item.get("rating") and float(item["rating"]) >= 4.0:
            positive_count += 1
        elif str(item.get("sentiment", "")).lower() in ("positive", "satisfied", "helpful"):
            positive_count += 1
        elif item.get("score") and float(item["score"]) > 0.6:
            positive_count += 1

    percent = (positive_count / total) * 100.0 if total > 0 else 0.0
    return {
        "positive": positive_count,
        "total": total,
        "percent": round(percent, 2)
    }


def _has_escalation(text: str) -> bool:
    """Check if text contains escalation phrases."""
    if not text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in ESCALATION_PHRASES)


def compute_task_completion_rate(message_records: List[Dict[str, Any]]) -> float:
    """
    Heuristic: message is 'completed' if response does not end with or contain escalation phrases.
    Escalation phrases: 'contact HR', 'contact IT', 'reach out to', 'please email', 'submit a ticket'
    """
    if not message_records:
        return 1.0

    completed_count = 0
    total = len(message_records)

    for msg in message_records:
        response_text = msg.get("response") or msg.get("content") or msg.get("answer") or ""
        # Check if response text escalated
        if not _has_escalation(response_text):
            completed_count += 1

    return round(float(completed_count / total), 4) if total > 0 else 1.0


def compute_escalation_rate(message_records: List[Dict[str, Any]]) -> float:
    """
    Fraction of responses containing escalation phrases.
    """
    if not message_records:
        return 0.0

    escalation_count = 0
    total = len(message_records)

    for msg in message_records:
        response_text = msg.get("response") or msg.get("content") or msg.get("answer") or ""
        if _has_escalation(response_text):
            escalation_count += 1

    return round(float(escalation_count / total), 4) if total > 0 else 0.0


def estimate_cost_per_task(message_records: List[Dict[str, Any]], cost_per_1k_tokens: float = 0.001) -> float:
    """
    Estimate cost based on token counts (from metadata or estimated from text length).
    """
    if not message_records:
        return 0.0

    total_tokens = 0
    for msg in message_records:
        meta = msg.get("metadata", {})
        prompt_tokens = meta.get("prompt_tokens")
        completion_tokens = meta.get("completion_tokens")
        total_tok = meta.get("total_tokens")

        if total_tok:
            total_tokens += int(total_tok)
        elif prompt_tokens or completion_tokens:
            total_tokens += int(prompt_tokens or 0) + int(completion_tokens or 0)
        else:
            # Fallback estimation: ~4 chars per token
            query_len = len(msg.get("query", "") or msg.get("question", ""))
            resp_len = len(msg.get("response", "") or msg.get("content", ""))
            total_tokens += int((query_len + resp_len) / 4)

    total_cost = (total_tokens / 1000.0) * cost_per_1k_tokens
    cost_per_task = total_cost / max(1, len(message_records))
    return round(float(cost_per_task), 6)
