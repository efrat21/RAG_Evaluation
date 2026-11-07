"""
Document ingestion service.
Exposes:
  POST /ingest - trigger ingestion of all docs in RAW_DATA_PATH
  GET /health - health check
  GET /metrics - Prometheus metrics
  GET /status - last ingestion status
  GET /quality-report - latest data quality report
"""

import os
import glob
import uuid
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app
from qdrant_client import QdrantClient, models
from langchain_ollama import OllamaEmbeddings

# Import OpenTelemetry components safely
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Import internal modules
import parsers
import chunker
import quality
import golden_dataset

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion_service")

# OTEL Setup
try:
    resource = Resource.create(attributes={"service.name": "knowledge-assistant-ingestion"})
    provider = TracerProvider(resource=resource)
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    except Exception as exp_err:
        logger.warning(f"OTLP exporter init skipped: {exp_err}")
    trace.set_tracer_provider(provider)
except Exception as otel_err:
    logger.warning(f"TracerProvider setup skipped: {otel_err}")

tracer = trace.get_tracer("ingestion_service")

# Configurations
def _resolve_path(env_var: str, default_docker: str, default_local: str) -> str:
    val = os.getenv(env_var)
    if val:
        return val
    if os.path.exists(default_local):
        return default_local
    return default_docker

RAW_DATA_PATH = _resolve_path(
    "RAW_DATA_PATH",
    "/data/raw",
    r"C:\Users\efrat\.gemini\antigravity\scratch\knowledge-assistant\data\raw"
)
GOLDEN_DATASET_PATH = _resolve_path(
    "GOLDEN_DATASET_PATH",
    "/data/golden_dataset/golden.jsonl",
    r"C:\Users\efrat\.gemini\antigravity\scratch\knowledge-assistant\data\golden_dataset\golden.jsonl"
)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "company_knowledge")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

app = FastAPI(title="Document Ingestion Service", version="1.0.0")

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Service state
ingestion_state: Dict[str, Any] = {
    "status": "idle",
    "last_run_start": None,
    "last_run_end": None,
    "last_report": None,
    "quality_reports": []
}


def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant client connection."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30.0)


def ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 768):
    """Ensure collection exists in Qdrant with cosine distance."""
    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        if not exists:
            logger.info(f"Creating Qdrant collection '{collection_name}' (vector size {vector_size})")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )
        else:
            logger.info(f"Qdrant collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Error checking/creating Qdrant collection: {e}")
        raise


async def run_ingestion() -> Dict[str, Any]:
    """
    Main ingestion flow:
    1. Scan RAW_DATA_PATH for all supported files
    2. For each file:
       a. Parse with parsers.parse_document()
       b. Chunk with chunker.chunk_document()
       c. Run quality checks with quality.run_quality_report()
       d. Embed chunks with OllamaEmbeddings
       e. Upsert to Qdrant with payload: {chunk_id, doc_id, content, title, section, source_path, doc_type, metadata}
    3. After all docs ingested, generate golden dataset with golden_dataset.generate_golden_dataset()
    4. Return ingestion report {docs_processed, chunks_created, failed, golden_qa_count}
    """
    with tracer.start_as_current_span("run_ingestion") as span:
        ingestion_state["status"] = "running"
        ingestion_state["last_run_start"] = datetime.now().isoformat()
        start_time = datetime.now()

        docs_processed = 0
        chunks_created = 0
        failed_docs = 0
        all_chunks: List[chunker.DocumentChunk] = []
        all_doc_ids: List[str] = []
        doc_quality_reports: List[Dict[str, Any]] = []

        logger.info(f"Starting ingestion scan in: {RAW_DATA_PATH}")

        # Connect to Qdrant
        qdrant_client = None
        try:
            qdrant_client = get_qdrant_client()
            ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION, vector_size=768)
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {e}")

        # Initialize Embeddings
        embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_URL
        )

        # 1. Scan RAW_DATA_PATH for all files
        supported_extensions = ("*.txt", "*.pdf", "*.docx", "*.doc", "*.md", "*.html", "*.htm", "*.py", "*.json")
        file_paths = []
        for ext in supported_extensions:
            file_paths.extend(glob.glob(os.path.join(RAW_DATA_PATH, ext)))
            file_paths.extend(glob.glob(os.path.join(RAW_DATA_PATH, "**", ext), recursive=True))

        file_paths = sorted(list(set(file_paths)))
        logger.info(f"Discovered {len(file_paths)} candidate files for ingestion.")

        # 2. Iterate each file
        for file_path in file_paths:
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, os.path.basename(file_path)))
            all_doc_ids.append(doc_id)

            try:
                with tracer.start_as_current_span("process_document") as doc_span:
                    doc_span.set_attribute("file.path", file_path)
                    doc_span.set_attribute("doc.id", doc_id)

                    # a. Parse document
                    parsed_doc = parsers.parse_document(file_path)
                    if parsed_doc.parse_errors and not parsed_doc.content:
                        failed_docs += 1
                        logger.error(f"Failed to parse {file_path}: {parsed_doc.parse_errors}")
                        continue

                    # b. Chunk document
                    chunks = chunker.chunk_document(parsed_doc, doc_id=doc_id)
                    all_chunks.extend(chunks)

                    # c. Quality report
                    q_report = await quality.run_quality_report(
                        doc=parsed_doc,
                        chunks=chunks,
                        qdrant_client=qdrant_client,
                        collection_name=QDRANT_COLLECTION
                    )
                    doc_quality_reports.append(q_report)

                    # d. Embed and upsert chunks to Qdrant
                    if chunks and qdrant_client:
                        try:
                            chunk_texts = [c.content for c in chunks]
                            # Batch embeddings
                            vectors = await embedder.aembed_documents(chunk_texts)

                            points = []
                            for idx, chunk in enumerate(chunks):
                                payload = {
                                    "chunk_id": chunk.chunk_id,
                                    "doc_id": chunk.doc_id,
                                    "content": chunk.content,
                                    "title": chunk.title,
                                    "section": chunk.section,
                                    "source_path": chunk.source_path,
                                    "doc_type": chunk.doc_type,
                                    "char_count": chunk.char_count,
                                    "is_complete": chunk.is_complete,
                                    "metadata": chunk.metadata,
                                }
                                points.append(
                                    models.PointStruct(
                                        id=chunk.chunk_id,
                                        vector=vectors[idx],
                                        payload=payload
                                    )
                                )

                            qdrant_client.upsert(
                                collection_name=QDRANT_COLLECTION,
                                points=points,
                                wait=True
                            )
                        except Exception as qdrant_err:
                            logger.error(f"Error vectorizing or upserting {file_path} to Qdrant: {qdrant_err}")

                    docs_processed += 1
                    chunks_created += len(chunks)
                    logger.info(f"Ingested {file_path}: {len(chunks)} chunks, quality score {parsed_doc.parse_quality_score}")

            except Exception as doc_err:
                failed_docs += 1
                logger.exception(f"Unhandled error ingesting {file_path}: {doc_err}")

        # Check coverage
        if qdrant_client and all_doc_ids:
            try:
                await quality.check_index_coverage(qdrant_client, QDRANT_COLLECTION, all_doc_ids)
            except Exception as cov_err:
                logger.warning(f"Index coverage check warning: {cov_err}")

        # 3. Generate golden dataset from ingested chunks
        golden_qa_count = 0
        try:
            with tracer.start_as_current_span("generate_golden_dataset"):
                logger.info(f"Starting golden dataset generation to: {GOLDEN_DATASET_PATH}")
                qa_records = await golden_dataset.generate_golden_dataset(
                    chunks=all_chunks,
                    llm_model=LLM_MODEL,
                    ollama_url=OLLAMA_URL,
                    output_path=GOLDEN_DATASET_PATH,
                    n_questions_per_chunk=3,
                    max_chunks=50,
                    min_chunk_length=200
                )
                golden_qa_count = len(qa_records)
        except Exception as golden_err:
            logger.error(f"Golden dataset generation error: {golden_err}")

        duration_sec = (datetime.now() - start_time).total_seconds()
        report = {
            "docs_processed": docs_processed,
            "chunks_created": chunks_created,
            "failed": failed_docs,
            "golden_qa_count": golden_qa_count,
            "duration_seconds": round(duration_sec, 2),
            "timestamp": datetime.now().isoformat()
        }

        span.set_attribute("ingestion.docs_processed", docs_processed)
        span.set_attribute("ingestion.chunks_created", chunks_created)
        span.set_attribute("ingestion.golden_qa_count", golden_qa_count)

        ingestion_state["status"] = "completed"
        ingestion_state["last_run_end"] = datetime.now().isoformat()
        ingestion_state["last_report"] = report
        ingestion_state["quality_reports"] = doc_quality_reports

        return report


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "document-ingestion",
        "raw_data_path": RAW_DATA_PATH,
        "golden_dataset_path": GOLDEN_DATASET_PATH,
        "qdrant_collection": QDRANT_COLLECTION,
    }


@app.get("/status")
def get_status():
    """Return last ingestion status and summary."""
    return ingestion_state


@app.get("/quality-report")
def get_quality_report():
    """Returns data quality summaries across ingested documents."""
    reports = ingestion_state.get("quality_reports", [])
    avg_quality = 0.0
    if reports:
        avg_quality = sum(r.get("parse_quality_score", 1.0) for r in reports) / len(reports)

    return {
        "total_documents_checked": len(reports),
        "average_parse_quality_score": round(avg_quality, 3),
        "reports": reports
    }


class IngestRequest(BaseModel):
    background: bool = False


@app.post("/ingest")
async def trigger_ingestion(
    background_tasks: BackgroundTasks,
    request: Optional[IngestRequest] = None,
):
    """Trigger ingestion of all docs in RAW_DATA_PATH."""
    if ingestion_state["status"] == "running":
        return {"status": "already_running", "message": "An ingestion task is currently running."}

    is_background = request.background if request else False

    if is_background:
        if background_tasks:
            background_tasks.add_task(run_ingestion)
        else:
            asyncio.create_task(run_ingestion())
        return {
            "status": "started",
            "message": "Ingestion process initiated in background."
        }
    else:
        report = await run_ingestion()
        return {
            "status": "completed",
            "report": report
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ingest:app", host="0.0.0.0", port=8001, reload=False)
