# 🧠 IntelliBase — Enterprise Internal Knowledge Assistant

A production-architected RAG system for enterprise internal knowledge management, featuring:
- **5-Layer Evaluation Framework** (Data Quality → Retrieval → Generation → Reliability → Business)
- **OpenTelemetry Distributed Tracing** (Jaeger + Prometheus + Grafana)
- **Guardrails** (prompt injection, PII leakage, off-topic, toxicity, data exfiltration)
- **RAGAS Integration** for retrieval and generation quality evaluation
- **Llama 3.1 (8B)** via Ollama (fully local, GPU auto-detected)
- **Rich Web UI** — glassmorphism chat interface + live monitoring dashboard

---

## Architecture

```
┌─────────────────────────────────────────────┐
│          Web UI (HTML/CSS/JS)                │
│   Chat (port 80)    Dashboard (Grafana)      │
└──────────────────┬──────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────┐
│          FastAPI Backend  :8000              │
│  Auth → Guard → Rewrite → Retrieve →        │
│  Rerank → Prompt → Generate → Guard         │
└──────┬───────────────────────────┬──────────┘
       │                           │
┌──────▼──────┐           ┌────────▼────────┐
│   Qdrant    │           │   Ollama         │
│  :6333      │           │  :11434          │
│ (vectors)   │           │  llama3.1:8b     │
└─────────────┘           │  nomic-embed-text│
                          └─────────────────┘
       │
┌──────▼────────────────────────────────────┐
│          OpenTelemetry Collector           │
└──────┬────────────────────┬───────────────┘
       ▼                    ▼
┌──────────────┐   ┌──────────────────────┐
│    Jaeger    │   │  Prometheus → Grafana │
│   :16686     │   │  :9090      :3000     │
└──────────────┘   └──────────────────────┘
```

---

## Prerequisites

- **Docker** ≥ 24.0 with Docker Compose V2
- **NVIDIA GPU** (recommended for Llama 70B; 8B runs on CPU but is slower)
  - NVIDIA Container Toolkit installed
- **RAM**: Minimum 16GB (32GB recommended for 70B)
- **Disk**: ~25GB free (models + data)

---

## Quick Start

### 1. Clone and configure

```bash
git clone <repo-url>
cd knowledge-assistant
cp .env.example .env
# Edit .env to change JWT_SECRET_KEY and passwords
```

### 2. Start all services

```bash
docker compose up -d
```

This will:
- Pull and start all Docker images (~5 min first run)
- Automatically download `llama3.1:8b` and `nomic-embed-text` models via Ollama
- Initialize Qdrant collection
- Pre-provision Grafana with the Knowledge Assistant dashboard

### 3. Ingest documents

```bash
# Trigger ingestion of all sample documents in data/raw/
curl.exe -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d "{\"background\":true}"
```

In PowerShell, you can instead run:

```powershell
Invoke-RestMethod -Uri http://localhost:8001/ingest `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"background":true}'
```

Watch the logs:
```bash
docker compose logs -f ingestion
```

Check progress in PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:8001/status
```

This will:
- Parse 15 synthetic company documents (HR, IT, Finance, Legal, Engineering)
- Chunk, embed, and index them in Qdrant
- Auto-generate a golden Q&A dataset for evaluation

### 4. Open the app

| Service | URL | Credentials |
|---|---|---|
| **Chat UI** | http://localhost/frontend/index.html | alice / password123 |
| **Dashboard** | http://localhost/frontend/dashboard.html | (same) |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Jaeger** | http://localhost:16686 | — |
| **Qdrant** | http://localhost:6333/dashboard | — |
| **API Docs** | http://localhost:8000/docs | — |

> **Note**: Serve the frontend by opening the HTML files directly in your browser,
> or run `python -m http.server 8080` in the `frontend/` directory.

---

## Project Structure

```
knowledge-assistant/
├── docker-compose.yml          # All 9 services
├── .env.example                # Configuration template
├── README.md
│
├── services/
│   ├── api/                    # FastAPI backend (RAG + auth + tracing)
│   │   ├── main.py             # App entry point, routes
│   │   ├── auth.py             # JWT authentication
│   │   ├── rag_pipeline.py     # LangGraph RAG state machine
│   │   ├── guardrails.py       # 5-threat guardrail system
│   │   ├── tracing.py          # OpenTelemetry setup + span helpers
│   │   ├── models.py           # Pydantic request/response models
│   │   ├── database.py         # SQLite (feedback, sessions, traces)
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   ├── ingestion/              # Document processing pipeline
│   │   ├── ingest.py           # FastAPI + orchestration
│   │   ├── parsers.py          # PDF, DOCX, MD, HTML, code parsers
│   │   ├── chunker.py          # Semantic recursive chunking
│   │   ├── quality.py          # Layer 1 quality checks + metrics
│   │   ├── golden_dataset.py   # LLM-based Q&A generation
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── evaluation/             # 5-layer evaluation framework
│       ├── evaluator.py        # FastAPI + evaluation orchestrator
│       ├── ragas_eval.py       # RAGAS (Layers 2 + 3)
│       ├── metrics.py          # Custom metrics (Layers 2, 4, 5)
│       ├── requirements.txt
│       └── Dockerfile
│
├── frontend/
│   ├── index.html              # Login + Chat UI
│   ├── dashboard.html          # Monitoring dashboard
│   ├── css/styles.css
│   └── js/
│       ├── chat.js
│       └── dashboard.js
│
├── data/
│   ├── raw/                    # 15 synthetic company documents
│   └── golden_dataset/
│       └── golden.jsonl        # Auto-generated after ingestion
│
├── otel/
│   ├── otel-collector.yml      # OTEL collector pipeline
│   └── prometheus.yml          # Prometheus scrape config
│
└── grafana/
    ├── provisioning/           # Auto-provisioned datasources + dashboards
    └── dashboards/
        └── knowledge-assistant.json  # Pre-built 5-layer dashboard
```

---

## 5-Layer Evaluation

Run evaluation on-demand from the Dashboard UI ("▶ Run Evaluation" button) or via API:

```bash
curl -X POST http://localhost:8002/evaluate
# Poll for results:
curl http://localhost:8002/evaluate/results
```

| Layer | Metrics | Tool |
|---|---|---|
| **1. Data Quality** | Chunk quality, freshness, parse errors, index coverage | Custom + Prometheus |
| **2. Retrieval** | Precision@5, Recall@5, MRR, NDCG@5, Context Precision, Context Recall | Custom + RAGAS |
| **3. Generation** | Faithfulness, Correctness, Relevance, Hallucination Rate | RAGAS |
| **4. Reliability** | P50/P95/P99 latency, error rate, guardrail blocks | OTEL + Prometheus |
| **5. Business** | CSAT (👍/👎), task completion, escalation rate, cost/task | SQLite + Custom |

---

## Tests

Run the Python unit tests (no Docker services required):

```bash
python -m unittest discover -s tests -v
```

Run the frontend controller tests with Node.js 18 or newer:

```bash
node --test tests/frontend.test.mjs
```

---

## Distributed Tracing

Every request creates a full distributed trace with 9 span types:

```
rag.request
├── guardrail.input       [threat_detected, threat_type, blocked]
├── query.rewrite         [original_query, rewritten_query]
├── retrieval.hybrid      [duration_ms, k, doc_ids[], scores[]]
├── retrieval.rerank      [rerank_duration_ms]
├── prompt.build          [prompt_version, context_size_tokens]
├── llm.generate          [model, duration_ms, tokens, finish_reason]
├── guardrail.output      [pii_detected, redacted_fields[]]
└── response.final        [status, eval_score, feedback]
```

View traces at **http://localhost:16686** → Service: `knowledge-assistant`

---

## Guardrails

| Threat | Method |
|---|---|
| **Prompt injection** | Regex patterns + LLM binary classifier |
| **PII leakage** | Microsoft Presidio (detect + redact in output) |
| **Off-topic queries** | Embedding cosine similarity to company topic centroid |
| **Toxic content** | Keyword blocklist + LLM classifier |
| **Data exfiltration** | Bulk query pattern detection + rate limiting |

---

## Demo Users

| Username | Password | Role |
|---|---|---|
| alice | password123 | Employee |
| bob | password456 | Employee |
| admin | adminpass | Admin |

Change these in `.env` before any real use.

---

## Useful Commands

```bash
# View all service logs
docker compose logs -f

# Restart a specific service
docker compose restart api

# Check Ollama models
curl http://localhost:11434/api/tags

# Manual ingestion
curl -X POST http://localhost:8001/ingest

# Run evaluation
curl -X POST http://localhost:8002/evaluate

# API health check
curl http://localhost:8000/health

# Stop everything
docker compose down

# Stop and remove all data volumes
docker compose down -v
```

---

## Configuration

All settings are in `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | Ollama model for generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `RETRIEVAL_TOP_K` | `5` | Documents retrieved per query |
| `GUARDRAIL_TOPIC_THRESHOLD` | `0.7` | Topic relevance threshold (0-1) |
| `JWT_EXPIRE_MINUTES` | `480` | Session duration (8 hours) |
| `DEMO_USERS` | `alice:password123,...` | Comma-separated user:pass pairs |

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.1 8B via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector DB | Qdrant (hybrid dense + BM25) |
| RAG Framework | LangChain + LangGraph |
| Backend | Python 3.11 + FastAPI |
| Auth | JWT (python-jose) |
| PII Detection | Microsoft Presidio |
| Evaluation | RAGAS + custom metrics |
| Tracing | OpenTelemetry → Jaeger |
| Metrics | Prometheus + Grafana |
| Storage | SQLite (feedback + sessions) |
| Frontend | Vanilla HTML/CSS/JS |
