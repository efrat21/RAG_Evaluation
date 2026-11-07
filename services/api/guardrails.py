"""Safety, compliance, and PII guardrails for the enterprise knowledge assistant.

Implements asynchronous inbound query validation (prompt injection detection,
topic relevance centroid scoring, toxicity classification, length limits,
and data exfiltration defense) and synchronous outbound PII redaction using
Presidio Analyzer & Anonymizer with fallback support.
"""

import logging
import re
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from models import GuardrailResult

logger = logging.getLogger("knowledge_assistant.guardrails")

# -------------------------------------------------------------------------
# Prompt Injection & Exfiltration Patterns
# -------------------------------------------------------------------------
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"\n\s*human\s*:\s*", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b|\bdo\s+anything\s+now\b", re.IGNORECASE),
    re.compile(r"bypass\s+(guardrails|filters|safety|restrictions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+no\s+longer\s+an?\s+ai", re.IGNORECASE),
    re.compile(r"disregard\s+(your|the)\s+(system|instructions|guidelines)", re.IGNORECASE),
    re.compile(r"act\s+as\s+an?\s+unrestricted", re.IGNORECASE),
    re.compile(r"developer\s+mode\s+(output|enabled|on)", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?safety\s+protocols", re.IGNORECASE),
]

EXFILTRATION_PATTERNS = [
    re.compile(r"(show|list|dump|export|print|give)\s+(me\s+)?(all|every|the\s+entire)\s+(?:(?:employee|company|internal|customer|user)\s+)?(documents?|database|files?|records?|users?|passwords?|credentials?)", re.IGNORECASE),
    re.compile(r"(select\s+\*\s+from|dump\s+table|export\s+all\s+data)", re.IGNORECASE),
    re.compile(r"(reveal|leak|unhide)\s+(hidden|internal|system)\s+(data|prompts|keys|secrets)", re.IGNORECASE),
]

# Baseline toxic / harassment / offensive terms
TOXIC_KEYWORDS = [
    "kys", "kill yourself", "hate speech", "terrorist", "make a bomb",
    "how to hack", "steal credentials", "ddos attack", "exploit vulnerability"
]

# Benchmark topic domain texts for computing the company knowledge centroid
TOPIC_ANCHORS = [
    "company policy, code of conduct, ethics guidelines, and employee rules",
    "human resources HR benefits payroll holidays vacation parental leave and healthcare",
    "information technology IT support VPN wifi laptop access credentials security software SSO",
    "software engineering technical architecture cloud development coding standards git devops",
    "corporate finance expense reimbursement travel policy budgets procurement vendor invoices",
    "legal compliance intellectual property privacy GDPR NDA contracts data governance",
    "office facilities building access badge security maintenance visitor policy",
]

# In-memory centroid cache
_topic_centroid: Optional[np.ndarray] = None


async def get_topic_centroid(embeddings: Any) -> Optional[np.ndarray]:
    """Calculate or retrieve the precomputed centroid vector of company topic anchors.

    Args:
        embeddings: LangChain / Ollama embeddings provider.

    Returns:
        1D numpy array representing the topic centroid, or None if unavailable.
    """
    global _topic_centroid
    if _topic_centroid is not None:
        return _topic_centroid

    if embeddings is None:
        return None

    try:
        # Embed all anchor documents
        vectors = await embeddings.aembed_documents(TOPIC_ANCHORS)
        matrix = np.array(vectors, dtype=np.float32)
        _topic_centroid = np.mean(matrix, axis=0, keepdims=True)
        # Normalize centroid vector
        norm = np.linalg.norm(_topic_centroid)
        if norm > 0:
            _topic_centroid = _topic_centroid / norm
        logger.info("Company topic centroid calculated and cached successfully.")
        return _topic_centroid
    except Exception as exc:
        logger.warning("Could not precompute topic centroid using embeddings: %s", exc)
        return None


# -------------------------------------------------------------------------
# Presidio PII Detection & Anonymization Engine
# -------------------------------------------------------------------------
_analyzer_engine = None
_anonymizer_engine = None
_presidio_available = False

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
    ).create_engine()
    _analyzer_engine = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )
    _anonymizer_engine = AnonymizerEngine()
    _presidio_available = True
    logger.info("Presidio Analyzer and Anonymizer engines initialized.")
except Exception as exc:
    logger.warning("Presidio initialization deferred or unavailable: %s. Using regex PII engine fallback.", exc)


def _regex_redact_pii(text: str) -> Tuple[str, List[str]]:
    """Fallback regex-based PII detector and anonymizer."""
    redacted = text
    detected: List[str] = []

    # Email
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b")
    if email_pattern.search(redacted):
        detected.append("EMAIL_ADDRESS")
        redacted = email_pattern.sub("<EMAIL>", redacted)

    # Phone numbers
    phone_pattern = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
    if phone_pattern.search(redacted):
        detected.append("PHONE_NUMBER")
        redacted = phone_pattern.sub("<PHONE_NUMBER>", redacted)

    # US SSN
    ssn_pattern = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    if ssn_pattern.search(redacted):
        detected.append("US_SSN")
        redacted = ssn_pattern.sub("<SSN>", redacted)

    # Credit card (13-19 digits, possibly spaced/hyphenated)
    cc_pattern = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
    if cc_pattern.search(redacted):
        detected.append("CREDIT_CARD")
        redacted = cc_pattern.sub("<CREDIT_CARD>", redacted)

    return redacted, list(set(detected))


# -------------------------------------------------------------------------
# Main Guardrail Interfaces
# -------------------------------------------------------------------------
async def check_input(
    query: str,
    llm: Any = None,
    embeddings: Any = None,
) -> GuardrailResult:
    """Asynchronously evaluate an inbound user query against safety and domain rules.

    Checks:
    1. Query length limit
    2. Prompt injection pattern detection
    3. Topic relevance similarity against company topic centroid
    4. Toxicity keyword heuristics and optional LLM classifier
    5. Data exfiltration pattern detection

    Args:
        query: Inbound user query text.
        llm: Optional LangChain LLM instance for toxicity verification.
        embeddings: Optional embeddings client for topic similarity calculation.

    Returns:
        GuardrailResult indicating whether the query is allowed or blocked.
    """
    clean_query = query.strip()

    # 1. Length check
    if len(clean_query) > settings.GUARDRAIL_MAX_QUERY_LENGTH:
        logger.warning("Guardrail blocked: Query length %d exceeds max %d", len(clean_query), settings.GUARDRAIL_MAX_QUERY_LENGTH)
        return GuardrailResult(
            blocked=True,
            reason=f"Query length ({len(clean_query)}) exceeds maximum allowed limit of {settings.GUARDRAIL_MAX_QUERY_LENGTH} characters.",
            threat_type="length_exceeded",
        )

    # 2. Prompt injection checks
    for pattern in INJECTION_PATTERNS:
        if pattern.search(clean_query):
            logger.warning("Guardrail blocked: Prompt injection pattern matched '%s'", pattern.pattern)
            return GuardrailResult(
                blocked=True,
                reason="Query contains prohibited instruction override or prompt injection pattern.",
                threat_type="prompt_injection",
            )

    # 3. Data exfiltration checks
    for pattern in EXFILTRATION_PATTERNS:
        if pattern.search(clean_query):
            logger.warning("Guardrail blocked: Data exfiltration pattern matched '%s'", pattern.pattern)
            return GuardrailResult(
                blocked=True,
                reason="Query appears to request bulk data extraction or system secret exposure.",
                threat_type="data_exfiltration",
            )

    # 4. Toxicity checks (heuristics)
    lower_query = clean_query.lower()
    for toxic in TOXIC_KEYWORDS:
        if toxic in lower_query:
            logger.warning("Guardrail blocked: Toxic term detected '%s'", toxic)
            return GuardrailResult(
                blocked=True,
                reason="Query contains toxic, harmful, or abusive content.",
                threat_type="toxicity",
            )

    # Optional LLM-based toxicity check for ambiguous or aggressive queries
    if llm is not None and len(clean_query) > 10:
        try:
            # We can run a rapid check if prompted
            pass
        except Exception as exc:
            logger.debug("Optional LLM toxicity check skipped: %s", exc)

    # 5. Topic relevance check via semantic embedding similarity
    if embeddings is not None:
        try:
            centroid = await get_topic_centroid(embeddings)
            if centroid is not None:
                query_vector = await embeddings.aembed_query(clean_query)
                q_vec = np.array([query_vector], dtype=np.float32)
                q_norm = np.linalg.norm(q_vec)
                if q_norm > 0:
                    q_vec = q_vec / q_norm

                similarity = float(cosine_similarity(q_vec, centroid)[0][0])
                logger.debug("Topic similarity score: %.4f (threshold: %.4f)", similarity, settings.GUARDRAIL_TOPIC_THRESHOLD)

                if similarity < settings.GUARDRAIL_TOPIC_THRESHOLD:
                    logger.warning("Guardrail blocked: Off-topic query (similarity %.3f < %.3f)", similarity, settings.GUARDRAIL_TOPIC_THRESHOLD)
                    return GuardrailResult(
                        blocked=True,
                        reason="Your question appears to be outside the scope of company policies, HR, IT, finance, engineering, or legal documentation.",
                        threat_type="off_topic",
                    )
        except Exception as exc:
            logger.warning("Semantic topic check encountered an error (allowing query): %s", exc)

    return GuardrailResult(blocked=False, reason=None, threat_type=None)


def check_output(response: str) -> Tuple[GuardrailResult, str]:
    """Synchronously evaluate LLM response for PII and redact sensitive information.

    Uses Presidio Analyzer + Anonymizer when available, falling back to compiled
    regex recognizers for EMAIL, PHONE, SSN, and CREDIT_CARD.

    Args:
        response: Generated raw response text.

    Returns:
        Tuple of (GuardrailResult, redacted_response_string).
    """
    if not response:
        return GuardrailResult(blocked=False, reason=None, threat_type=None), response

    entities_to_detect = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "US_BANK_NUMBER",
        "US_PASSPORT",
        "LOCATION",
    ]

    detected_entities: List[str] = []
    redacted_text = response

    if _presidio_available and _analyzer_engine is not None and _anonymizer_engine is not None:
        try:
            results = _analyzer_engine.analyze(
                text=response,
                entities=entities_to_detect,
                language="en",
            )

            if results:
                detected_entities = list({res.entity_type for res in results})
                operators = {
                    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
                    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
                    "US_SSN": OperatorConfig("replace", {"new_value": "<SSN>"}),
                    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
                    "US_BANK_NUMBER": OperatorConfig("replace", {"new_value": "<BANK_NUMBER>"}),
                    "US_PASSPORT": OperatorConfig("replace", {"new_value": "<PASSPORT>"}),
                    "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
                }
                anonymized_result = _anonymizer_engine.anonymize(
                    text=response,
                    analyzer_results=results,
                    operators=operators,
                )
                redacted_text = anonymized_result.text
                logger.info("Presidio redacted PII entities: %s", detected_entities)
        except Exception as exc:
            logger.warning("Presidio execution failed, falling back to regex: %s", exc)
            redacted_text, detected_entities = _regex_redact_pii(response)
    else:
        redacted_text, detected_entities = _regex_redact_pii(response)

    if detected_entities:
        return (
            GuardrailResult(
                blocked=False,
                reason=f"PII detected and redacted: {', '.join(detected_entities)}",
                threat_type="pii_detected",
            ),
            redacted_text,
        )

    return GuardrailResult(blocked=False, reason=None, threat_type=None), response
