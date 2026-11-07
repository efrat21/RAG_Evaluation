"""LangGraph RAG pipeline with OpenTelemetry tracing, hybrid search, and guardrails.

Orchestrates multi-stage query lifecycle:
1. Inbound safety & topic guardrail validation
2. HyDE (Hypothetical Document Embeddings) query expansion/rewriting
3. Hybrid dense (Qdrant) + sparse (BM25) retrieval with Reciprocal Rank Fusion (RRF)
4. LLM cross-encoder reranking
5. Contextual prompt assembly with source attribution
6. LLM answer synthesis
7. Outbound PII scanning and redaction guardrails
"""

from datetime import datetime, timezone
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from config import settings
from guardrails import check_input, check_output
from models import GuardrailResult, SourceDocument
from tracing import SpanAttr, create_rag_span, get_tracer

logger = logging.getLogger("knowledge_assistant.rag_pipeline")


# -------------------------------------------------------------------------
# State Definition
# -------------------------------------------------------------------------
class RAGState(TypedDict):
    """Execution state passed through the LangGraph RAG pipeline."""

    request_id: str
    session_id: str
    query: str
    rewritten_query: str
    retrieved_docs: List[dict]  # [{doc_id, content, title, source_path, score, page_number, doc_type}]
    reranked_docs: List[dict]
    context: str
    response: str
    sources: List[dict]
    confidence_score: float
    guardrail_input: GuardrailResult
    guardrail_output: GuardrailResult
    error: Optional[str]
    trace_context: dict


# -------------------------------------------------------------------------
# Default Enterprise Seed Knowledge Base (for cold-start & fallback)
# -------------------------------------------------------------------------
DEFAULT_DOCUMENTS = [
    {
        "doc_id": "kb-policy-001",
        "title": "Employee Remote Work & Hybrid Schedule Policy",
        "content": (
            "Employees are eligible for hybrid remote work up to 3 days per week with manager approval. "
            "Core collaboration hours are 10:00 AM to 4:00 PM local time. Remote workers must have a secure "
            "high-speed internet connection and use the company corporate VPN at all times. Home office equipment "
            "stipend of $500 is provided upon onboarding for ergonomic desks and chairs."
        ),
        "source_path": "/policies/hr/remote_work_policy.pdf",
        "doc_type": "policy",
        "page_number": 1,
    },
    {
        "doc_id": "kb-policy-002",
        "title": "Corporate Travel and Expense Reimbursement Guidelines",
        "content": (
            "All business travel expenses must be pre-approved by department leads. Daily meal allowance (per diem) "
            "is capped at $75 domestic and $110 international. Itemized receipts are strictly required for all "
            "expenses exceeding $25. Expense reports must be submitted through the portal within 30 days of travel."
        ),
        "source_path": "/policies/finance/travel_expense_policy.pdf",
        "doc_type": "policy",
        "page_number": 2,
    },
    {
        "doc_id": "kb-it-003",
        "title": "IT Security and Multi-Factor Authentication (MFA) Setup",
        "content": (
            "Multi-Factor Authentication (MFA) via Okta Verify or hardware security keys is mandatory for accessing "
            "internal resources including email, Git repositories, AWS consoles, and Slack. Passwords must be at "
            "least 16 characters and changed every 90 days. Report lost or compromised devices immediately to it-security@company.com."
        ),
        "source_path": "/it/security/mfa_guide.pdf",
        "doc_type": "it_guide",
        "page_number": 1,
    },
    {
        "doc_id": "kb-hr-004",
        "title": "Annual Paid Time Off (PTO) and Parental Leave Benefits",
        "content": (
            "Full-time employees receive 20 days of paid vacation annually, accrued bi-weekly, plus 10 paid corporate "
            "holidays and 5 floating wellness days. Primary caregivers are entitled to 16 weeks of fully paid parental "
            "leave, and secondary caregivers receive 8 weeks. Unused vacation days up to 5 days roll over to the next year."
        ),
        "source_path": "/policies/hr/benefits_handbook.pdf",
        "doc_type": "benefits",
        "page_number": 4,
    },
    {
        "doc_id": "kb-eng-005",
        "title": "Engineering Production Deployment and Incident Management",
        "content": (
            "Production deployments occur Tuesdays and Thursdays between 10:00 AM and 2:00 PM UTC following green CI/CD "
            "pipeline runs and peer code review approvals from at least two senior engineers. Severity 1 outages require "
            "immediate assembly of the Incident Command team on the #sev-1-incident Slack channel with updates posted every 15 minutes."
        ),
        "source_path": "/engineering/runbooks/deployment_and_incidents.md",
        "doc_type": "engineering",
        "page_number": 3,
    },
    {
        "doc_id": "kb-legal-006",
        "title": "Data Privacy and Customer Information Confidentiality",
        "content": (
            "Customer data is classified as Confidential Level 3 and must be encrypted both in transit (TLS 1.3) "
            "and at rest (AES-256). Production data must never be copied to local workstations or non-production environments. "
            "Any suspected data breach or unauthorized disclosure must be reported within 1 hour to legal-privacy@company.com."
        ),
        "source_path": "/legal/data_privacy_charter.pdf",
        "doc_type": "legal",
        "page_number": 1,
    },
]


# -------------------------------------------------------------------------
# Component Clients & Index State
# -------------------------------------------------------------------------
_llm_client: Optional[ChatOllama] = None
_embeddings_client: Optional[OllamaEmbeddings] = None
_qdrant_client: Optional[QdrantClient] = None
_bm25_index: Optional[BM25Okapi] = None
_document_corpus: List[dict] = list(DEFAULT_DOCUMENTS)


def get_llm() -> ChatOllama:
    """Return singleton instance of ChatOllama."""
    global _llm_client
    if _llm_client is None:
        _llm_client = ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            num_predict=settings.LLM_MAX_TOKENS,
            timeout=30.0,
        )
    return _llm_client


def get_embeddings() -> OllamaEmbeddings:
    """Return singleton instance of OllamaEmbeddings."""
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.EMBEDDING_MODEL,
        )
    return _embeddings_client


def get_qdrant_client() -> QdrantClient:
    """Return singleton instance of Qdrant client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=5.0,
        )
    return _qdrant_client


def get_bm25_index() -> Tuple[BM25Okapi, List[dict]]:
    """Return in-memory BM25 index and associated document corpus."""
    global _bm25_index, _document_corpus
    if _bm25_index is None:
        tokenized_corpus = [doc["content"].lower().split() for doc in _document_corpus]
        _bm25_index = BM25Okapi(tokenized_corpus)
    return _bm25_index, _document_corpus


def initialize_qdrant_collection() -> None:
    """Ensure the Qdrant knowledge collection exists and index seed docs if needed."""
    try:
        client = get_qdrant_client()
        collections_response = client.get_collections()
        existing = [c.name for c in collections_response.collections]
        if settings.QDRANT_COLLECTION not in existing:
            logger.info("Creating Qdrant collection '%s' (dim: %d)", settings.QDRANT_COLLECTION, settings.EMBEDDING_DIM)
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )
    except Exception as exc:
        logger.warning("Could not initialize Qdrant collection (will rely on hybrid fallback): %s", exc)


# -------------------------------------------------------------------------
# Node 1: Input Guard Node
# -------------------------------------------------------------------------
async def input_guard_node(state: RAGState) -> Dict[str, Any]:
    """Validate inbound query through safety, prompt injection, and topic guardrails."""
    tracer = get_tracer()
    start_time = time.perf_counter()

    with create_rag_span(
        tracer,
        "rag.input_guard",
        query=state["query"],
        request_id=state["request_id"],
        session_id=state["session_id"],
    ) as span:
        guard_result = await check_input(
            query=state["query"],
            llm=get_llm(),
            embeddings=get_embeddings(),
        )

        duration_ms = (time.perf_counter() - start_time) * 1000
        span.set_attribute(SpanAttr.TOOL_DURATION_MS, duration_ms)
        span.set_attribute(SpanAttr.GUARDRAIL_BLOCKED, guard_result.blocked)
        if guard_result.threat_type:
            span.set_attribute(SpanAttr.GUARDRAIL_THREAT_TYPE, guard_result.threat_type)

        if guard_result.blocked:
            logger.warning("Input guardrail blocked query '%s': %s", state["query"][:50], guard_result.reason)
            return {
                "guardrail_input": guard_result,
                "error": guard_result.reason,
                "response": f"Your request cannot be processed: {guard_result.reason}",
            }

        return {
            "guardrail_input": guard_result,
            "error": None,
        }


# -------------------------------------------------------------------------
# Node 2: Query Rewrite Node (HyDE)
# -------------------------------------------------------------------------
async def query_rewrite_node(state: RAGState) -> Dict[str, Any]:
    """Expand and rewrite the user query using Hypothetical Document Embeddings (HyDE)."""
    tracer = get_tracer()
    query = state["query"]

    with create_rag_span(
        tracer,
        "rag.query_rewrite",
        query=query,
        request_id=state["request_id"],
    ) as span:
        llm = get_llm()
        hyde_prompt = (
            f"You are an enterprise knowledge search assistant. Write a short, hypothetical internal document excerpt "
            f"that provides a clear, authoritative answer to the following question. Do not include introductory notes or "
            f"disclaimers; write only the factual paragraph itself:\n\nQuestion: {query}\n\nPassage:"
        )

        try:
            hyde_response = await llm.ainvoke(hyde_prompt)
            hypothetical_answer = hyde_response.content.strip()
            if hypothetical_answer and len(hypothetical_answer) > 20:
                rewritten = f"{query} {hypothetical_answer[:300]}"
            else:
                rewritten = query
        except Exception as exc:
            logger.warning("HyDE rewrite failed (%s), defaulting to original query.", exc)
            rewritten = query

        span.set_attribute(SpanAttr.QUERY, query)
        span.set_attribute(SpanAttr.REWRITTEN_QUERY, rewritten)

        return {"rewritten_query": rewritten}


# -------------------------------------------------------------------------
# Node 3: Hybrid Retrieval Node (Dense Qdrant + Sparse BM25 + RRF)
# -------------------------------------------------------------------------
async def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Execute hybrid retrieval combining dense semantic search and BM25 sparse matching."""
    tracer = get_tracer()
    rewritten_query = state.get("rewritten_query") or state["query"]
    start_time = time.perf_counter()

    with create_rag_span(
        tracer,
        "rag.retrieval",
        query=state["query"],
        rewritten_query=rewritten_query,
    ) as span:
        dense_results: List[dict] = []
        sparse_results: List[dict] = []

        # 1. Dense Semantic Retrieval via Qdrant
        try:
            embeddings_model = get_embeddings()
            query_vector = await embeddings_model.aembed_query(rewritten_query)
            client = get_qdrant_client()

            search_response = client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=settings.RETRIEVAL_TOP_K,
                score_threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
            )

            for hit in search_response:
                payload = hit.payload or {}
                dense_results.append({
                    "doc_id": str(payload.get("doc_id", hit.id)),
                    "title": payload.get("title", "Internal Document"),
                    "content": payload.get("content", ""),
                    "source_path": payload.get("source_path", "/docs/general"),
                    "page_number": payload.get("page_number"),
                    "doc_type": payload.get("doc_type", "document"),
                    "score": float(hit.score),
                })
        except Exception as exc:
            logger.warning("Dense retrieval encountered an issue: %s. Relying on BM25.", exc)

        # 2. Sparse Retrieval via BM25
        try:
            bm25, corpus = get_bm25_index()
            tokens = rewritten_query.lower().split()
            bm25_scores = bm25.get_scores(tokens)

            # Sort indices descending
            top_indices = np.argsort(bm25_scores)[::-1][: settings.RETRIEVAL_TOP_K]
            for idx in top_indices:
                score = float(bm25_scores[idx])
                if score > 0.1 or not dense_results:
                    doc = corpus[idx]
                    sparse_results.append({
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "content": doc["content"],
                        "source_path": doc["source_path"],
                        "page_number": doc.get("page_number"),
                        "doc_type": doc.get("doc_type", "document"),
                        "score": score,
                    })
        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s", exc)

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score = 1 / (60 + rank_dense) + 1 / (60 + rank_sparse)
        rrf_k = 60.0
        doc_map: Dict[str, dict] = {}
        rrf_scores: Dict[str, float] = {}

        for rank, doc in enumerate(dense_results, start=1):
            did = doc["doc_id"]
            doc_map[did] = doc
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + rank))

        for rank, doc in enumerate(sparse_results, start=1):
            did = doc["doc_id"]
            if did not in doc_map:
                doc_map[did] = doc
            rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (rrf_k + rank))

        # Fallback if both dense and sparse yielded empty
        if not rrf_scores:
            logger.info("Hybrid retrieval empty. Using baseline corpus fallback.")
            for rank, doc in enumerate(_document_corpus[: settings.RETRIEVAL_TOP_K], start=1):
                did = doc["doc_id"]
                doc_map[did] = doc
                rrf_scores[did] = 1.0 / (rrf_k + rank)

        # Sort combined results by RRF score
        sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        retrieved_docs: List[dict] = []
        for did in sorted_doc_ids[: settings.RETRIEVAL_TOP_K]:
            item = dict(doc_map[did])
            # Normalize score into a friendly 0.0-1.0 range
            item["score"] = round(min(1.0, rrf_scores[did] * 30.0), 3)
            retrieved_docs.append(item)

        duration_ms = (time.perf_counter() - start_time) * 1000
        span.set_attribute(SpanAttr.RETRIEVAL_DURATION_MS, duration_ms)
        span.set_attribute(SpanAttr.NUM_DOCS_RETRIEVED, len(retrieved_docs))
        span.set_attribute(SpanAttr.DOC_IDS, [d["doc_id"] for d in retrieved_docs])
        span.set_attribute(SpanAttr.RELEVANCE_SCORES, [float(d["score"]) for d in retrieved_docs])

        logger.info("Retrieved %d documents in %.2f ms", len(retrieved_docs), duration_ms)
        return {"retrieved_docs": retrieved_docs}


# -------------------------------------------------------------------------
# Node 4: Cross-scoring Reranker Node
# -------------------------------------------------------------------------
async def rerank_node(state: RAGState) -> Dict[str, Any]:
    """Score and rerank retrieved documents using LLM cross-scoring."""
    tracer = get_tracer()
    query = state["query"]
    retrieved = state.get("retrieved_docs", [])

    if not retrieved:
        return {"reranked_docs": []}

    with create_rag_span(
        tracer,
        "rag.rerank",
        num_input_docs=len(retrieved),
    ) as span:
        llm = get_llm()
        scored_docs: List[dict] = []

        for doc in retrieved:
            scoring_prompt = (
                f"Rate the relevance of the following document excerpt to the user question on a scale from 0 to 10, "
                f"where 10 is perfectly relevant and answers the question directly, and 0 is completely irrelevant.\n"
                f"Output ONLY a single numeric score (0 to 10).\n\n"
                f"Question: {query}\n\n"
                f"Document: {doc['content'][:400]}\n\nScore:"
            )
            score_val = doc.get("score", 0.5)
            try:
                score_res = await llm.ainvoke(scoring_prompt)
                raw_text = score_res.content.strip().split()[0]
                extracted = float("".join(c for c in raw_text if c.isdigit() or c == "."))
                score_val = max(0.0, min(10.0, extracted)) / 10.0
            except Exception:
                # Retain original retrieval score on timeout or parsing issue
                pass

            doc_copy = dict(doc)
            doc_copy["score"] = round(float(score_val), 3)
            scored_docs.append(doc_copy)

        # Sort descending and retain top 3
        scored_docs.sort(key=lambda d: d["score"], reverse=True)
        top_reranked = scored_docs[:3]

        span.set_attribute(SpanAttr.DOC_IDS, [d["doc_id"] for d in top_reranked])
        span.set_attribute(SpanAttr.RELEVANCE_SCORES, [float(d["score"]) for d in top_reranked])

        return {"reranked_docs": top_reranked}


# -------------------------------------------------------------------------
# Node 5: Build Prompt Node
# -------------------------------------------------------------------------
async def build_prompt_node(state: RAGState) -> Dict[str, Any]:
    """Assemble the system instructions, citations context, and user prompt."""
    tracer = get_tracer()
    reranked = state.get("reranked_docs", [])
    query = state["query"]

    with create_rag_span(tracer, "rag.build_prompt") as span:
        context_blocks = []
        for idx, doc in enumerate(reranked, start=1):
            title = doc.get("title", "Document")
            content = doc.get("content", "").strip()
            context_blocks.append(f"[{idx}] {title}:\n{content}")

        context_str = "\n\n".join(context_blocks)
        token_estimate = int(len(context_str.split()) * 1.3)

        span.set_attribute(SpanAttr.PROMPT_VERSION, "v1.0")
        span.set_attribute(SpanAttr.CONTEXT_SIZE_TOKENS, token_estimate)

        return {"context": context_str}


# -------------------------------------------------------------------------
# Node 6: Answer Generation Node
# -------------------------------------------------------------------------
async def generate_node(state: RAGState) -> Dict[str, Any]:
    """Generate final synthesis using Ollama LLM and compute confidence score."""
    tracer = get_tracer()
    context = state.get("context", "")
    query = state["query"]
    reranked = state.get("reranked_docs", [])
    start_time = time.perf_counter()

    with create_rag_span(
        tracer,
        "rag.generate",
        llm_model=settings.LLM_MODEL,
    ) as span:
        system_instruction = (
            "System: You are an expert internal knowledge assistant for the company. "
            "Answer questions using ONLY the provided context. "
            "If the context doesn't contain the answer, say so explicitly and advise contacting the responsible team. "
            "Always cite your sources using bracketed notations like [1], [2] corresponding to the provided context."
        )

        user_content = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_content),
        ]

        llm = get_llm()
        duration_ms = 0.0
        response_text = ""
        prompt_tokens_est = int((len(system_instruction.split()) + len(user_content.split())) * 1.3)
        completion_tokens_est = 0

        try:
            ai_message = await llm.ainvoke(messages)
            response_text = ai_message.content.strip()
            duration_ms = (time.perf_counter() - start_time) * 1000
            completion_tokens_est = int(len(response_text.split()) * 1.3)

            span.set_attribute(SpanAttr.LLM_MODEL, settings.LLM_MODEL)
            span.set_attribute(SpanAttr.LLM_DURATION_MS, duration_ms)
            span.set_attribute(SpanAttr.LLM_PROMPT_TOKENS, prompt_tokens_est)
            span.set_attribute(SpanAttr.LLM_COMPLETION_TOKENS, completion_tokens_est)
            span.set_attribute(SpanAttr.LLM_STATUS, "success")
            span.set_attribute(SpanAttr.LLM_FINISH_REASON, "stop")
        except Exception as exc:
            logger.error("LLM answer generation failed: %s", exc)
            duration_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute(SpanAttr.LLM_STATUS, "error")
            span.set_attribute(SpanAttr.LLM_DURATION_MS, duration_ms)
            return {
                "error": str(exc),
                "response": "I encountered an error generating an answer. Please try again or contact IT support.",
            }

        # Calculate confidence score from top documents
        if reranked:
            scores = [float(d.get("score", 0.5)) for d in reranked]
            confidence = round(float(np.mean(scores)), 2)
        else:
            confidence = 0.2

        # Format sources into SourceDocument schema dictionaries
        sources_list: List[dict] = []
        for d in reranked:
            content_snippet = d.get("content", "")
            excerpt = content_snippet[:250] + "..." if len(content_snippet) > 250 else content_snippet
            sources_list.append({
                "doc_id": d["doc_id"],
                "title": d["title"],
                "excerpt": excerpt,
                "source_path": d.get("source_path", "/docs"),
                "relevance_score": float(d.get("score", 0.5)),
                "page_number": d.get("page_number"),
                "doc_type": d.get("doc_type", "document"),
            })

        return {
            "response": response_text,
            "sources": sources_list,
            "confidence_score": confidence,
            "error": None,
        }


# -------------------------------------------------------------------------
# Node 7: Output Guard Node (PII Redaction)
# -------------------------------------------------------------------------
async def output_guard_node(state: RAGState) -> Dict[str, Any]:
    """Scan generated response for PII entities and redact where detected."""
    tracer = get_tracer()
    raw_response = state.get("response", "")

    with create_rag_span(tracer, "rag.output_guard") as span:
        guard_result, anonymized_text = check_output(raw_response)

        span.set_attribute(SpanAttr.PII_DETECTED, bool(guard_result.threat_type == "pii_detected"))
        if guard_result.threat_type:
            span.set_attribute(SpanAttr.GUARDRAIL_THREAT_TYPE, guard_result.threat_type)

        return {
            "response": anonymized_text,
            "guardrail_output": guard_result,
        }


# -------------------------------------------------------------------------
# Terminal & Routing Nodes
# -------------------------------------------------------------------------
async def end_blocked_node(state: RAGState) -> Dict[str, Any]:
    """Terminal state for requests intercepted by guardrails."""
    reason = state.get("guardrail_input", GuardrailResult(blocked=True)).reason or "Request blocked by safety policy"
    return {
        "response": f"I cannot process your request: {reason}",
        "confidence_score": 0.0,
        "sources": [],
    }


async def end_error_node(state: RAGState) -> Dict[str, Any]:
    """Terminal state when generation or retrieval raises an unrecoverable error."""
    err = state.get("error") or "Unknown system error"
    return {
        "response": f"An error occurred while answering your request: {err}",
        "confidence_score": 0.0,
        "sources": [],
    }


def route_after_guard(state: RAGState) -> str:
    """Conditional router checking if input was blocked."""
    guard = state.get("guardrail_input")
    if guard and guard.blocked:
        return "end_blocked"
    return "query_rewrite"


def route_after_generate(state: RAGState) -> str:
    """Conditional router checking for generation errors."""
    if state.get("error"):
        return "end_error"
    return "output_guard"


# -------------------------------------------------------------------------
# Graph Construction & Compilation
# -------------------------------------------------------------------------
def create_rag_graph():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(RAGState)

    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("build_prompt", build_prompt_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("output_guard", output_guard_node)
    workflow.add_node("end_blocked", end_blocked_node)
    workflow.add_node("end_error", end_error_node)

    workflow.set_entry_point("input_guard")
    workflow.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"end_blocked": "end_blocked", "query_rewrite": "query_rewrite"},
    )
    workflow.add_edge("query_rewrite", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "build_prompt")
    workflow.add_edge("build_prompt", "generate")
    workflow.add_conditional_edges(
        "generate",
        route_after_generate,
        {"end_error": "end_error", "output_guard": "output_guard"},
    )
    workflow.add_edge("output_guard", END)
    workflow.add_edge("end_blocked", END)
    workflow.add_edge("end_error", END)

    return workflow.compile()


# Compile graph singleton
rag_application = create_rag_graph()


# -------------------------------------------------------------------------
# Public Execution Interface
# -------------------------------------------------------------------------
async def run_pipeline(query: str, session_id: str, request_id: str) -> Dict[str, Any]:
    """Execute the full LangGraph RAG pipeline for a given query.

    Args:
        query: Inbound user prompt.
        session_id: Active session identifier.
        request_id: Distributed trace request identifier.

    Returns:
        Dictionary containing response, sources, confidence_score, guardrails, and latency.
    """
    start_time = time.perf_counter()

    initial_state: RAGState = {
        "request_id": request_id,
        "session_id": session_id,
        "query": query,
        "rewritten_query": query,
        "retrieved_docs": [],
        "reranked_docs": [],
        "context": "",
        "response": "",
        "sources": [],
        "confidence_score": 0.0,
        "guardrail_input": GuardrailResult(blocked=False),
        "guardrail_output": GuardrailResult(blocked=False),
        "error": None,
        "trace_context": {},
    }

    try:
        final_state = await rag_application.ainvoke(initial_state)
    except Exception as exc:
        logger.exception("RAG pipeline execution failed unexpectedly: %s", exc)
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "message_id": str(uuid4()),
            "response": f"Pipeline failure: {str(exc)}",
            "sources": [],
            "confidence_score": 0.0,
            "guardrail": GuardrailResult(blocked=True, reason=str(exc), threat_type="internal_error"),
            "latency_ms": round(total_latency_ms, 2),
            "session_id": session_id,
            "request_id": request_id,
        }

    total_latency_ms = (time.perf_counter() - start_time) * 1000
    guard_input = final_state.get("guardrail_input", GuardrailResult(blocked=False))
    guard_output = final_state.get("guardrail_output", GuardrailResult(blocked=False))

    # The effective guardrail returned reflects either blocked input or detected output PII
    effective_guardrail = guard_input if guard_input.blocked else guard_output

    sources = [SourceDocument(**s) if isinstance(s, dict) else s for s in final_state.get("sources", [])]

    return {
        "message_id": str(uuid4()),
        "response": final_state.get("response", ""),
        "sources": sources,
        "confidence_score": float(final_state.get("confidence_score", 0.0)),
        "guardrail": effective_guardrail,
        "latency_ms": round(total_latency_ms, 2),
        "session_id": session_id,
        "request_id": request_id,
    }
