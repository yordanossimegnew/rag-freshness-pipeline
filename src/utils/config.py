"""Configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _pg(url: str, name: str) -> str:
    if not url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        raise ValueError(f"{name} must be a PostgreSQL URL.")
    return url


@dataclass(frozen=True)
class Settings:
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_freshness"       # pipeline state
    index_database_url: str = "postgresql://postgres:postgres@localhost:5432/docqa"         # Document QA's index
    docqa_path: Path = PROJECT_ROOT.parent / "document-qa-citations"
    model: str = "claude-opus-5-5"     # answers
    effort: str = "medium"
    top_k: int = 8                     # chunks sent to Claude per question
    recency_weight: float = 0.15       # how much a recently changed page is boosted in retrieval
    voyage_model: str = "voyage-3.5"
    embedding_dim: int = 1024
    chunk_chars: int = 2400            # keep equal to Document QA's settings so chunks match its own ingestion
    chunk_overlap: int = 300
    user_agent: str = "rag-freshness-pipeline/1.0 (+https://github.com/yordanossimegnew/rag-freshness-pipeline)"
    request_timeout: float = 20.0
    max_concurrency: int = 4
    per_host_delay: float = 0.5        # seconds between requests to the same host
    respect_robots: bool = True
    allow_private_urls: bool = False   # SSRF guard (tests/demo against localhost set this to true)
    max_page_bytes: int = 10 * 1024 * 1024
    keep_versions: int = 20            # text snapshots kept per page
    scheduler_enabled: bool = True
    scheduler_tick_seconds: float = 30.0
    failure_alert_threshold: int = 3   # consecutive failed syncs before a source alert
    alert_webhook_url: str | None = None
    voyage_embed_price: float = 0.0    # USD per million tokens, for the cost column
    log_level: str = "INFO"
    log_dir: Path = PROJECT_ROOT / "logs"

    def __post_init__(self) -> None:
        _pg(self.database_url, "DATABASE_URL")
        _pg(self.index_database_url, "INDEX_DATABASE_URL")


def load_settings(**overrides) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    docqa = os.getenv("DOCQA_PATH")
    s = Settings(
        database_url=os.getenv("DATABASE_URL", Settings.database_url),
        index_database_url=os.getenv("INDEX_DATABASE_URL", Settings.index_database_url),
        docqa_path=(PROJECT_ROOT / docqa).resolve() if docqa and not Path(docqa).is_absolute()
        else Path(docqa) if docqa else Settings.docqa_path,
        model=os.getenv("AGENT_MODEL", Settings.model),
        effort=os.getenv("AGENT_EFFORT", Settings.effort),
        voyage_model=os.getenv("VOYAGE_MODEL", Settings.voyage_model),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", Settings.embedding_dim)),
        chunk_chars=int(os.getenv("CHUNK_CHARS", Settings.chunk_chars)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", Settings.chunk_overlap)),
        max_concurrency=int(os.getenv("MAX_CONCURRENCY", Settings.max_concurrency)),
        respect_robots=_bool("RESPECT_ROBOTS", Settings.respect_robots),
        allow_private_urls=_bool("ALLOW_PRIVATE_URLS", Settings.allow_private_urls),
        scheduler_enabled=_bool("SCHEDULER_ENABLED", Settings.scheduler_enabled),
        scheduler_tick_seconds=float(os.getenv("SCHEDULER_TICK_SECONDS", Settings.scheduler_tick_seconds)),
        alert_webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        voyage_embed_price=float(os.getenv("VOYAGE_EMBED_PRICE_PER_MTOK", Settings.voyage_embed_price)),
        log_level=os.getenv("LOG_LEVEL", Settings.log_level),
    )
    return replace(s, **{k: v for k, v in overrides.items() if v is not None})
