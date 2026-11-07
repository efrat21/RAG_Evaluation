"""Configuration settings for the enterprise knowledge assistant API service.

Loads environment variables using Pydantic Settings with sensible defaults
for containerized and local execution.
"""

from typing import Dict, List
import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application metadata
    APP_NAME: str = "Enterprise Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "production"
    DEBUG: bool = False

    # Security & JWT
    JWT_SECRET_KEY: str = "rag-super-secret-jwt-key-for-enterprise-knowledge-assistant-2026"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 480
    DEMO_USERS: str = "alice:password123,bob:password456,admin:adminpass"

    # Ollama / LLM & Embeddings
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "llama3.1:8b"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # Qdrant Vector DB
    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "company_knowledge"
    EMBEDDING_DIM: int = 768
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.3

    # OpenTelemetry Tracing
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    OTEL_SERVICE_NAME: str = "knowledge-assistant"

    # Guardrails
    GUARDRAIL_MAX_QUERY_LENGTH: int = 1000
    GUARDRAIL_TOPIC_THRESHOLD: float = 0.7

    # Database
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", "knowledge_assistant.db")

    # CORS
    CORS_ORIGINS: str = "*"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        if self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    def parse_demo_users(self) -> Dict[str, str]:
        """Parse DEMO_USERS string 'user:pass,user2:pass2' into a dictionary."""
        users: Dict[str, str] = {}
        if not self.DEMO_USERS:
            return users
        entries = self.DEMO_USERS.split(",")
        for entry in entries:
            entry = entry.strip()
            if ":" in entry:
                username, password = entry.split(":", 1)
                users[username.strip()] = password.strip()
        return users


settings = Settings()
