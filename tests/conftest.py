from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent import build_app  # noqa: E402
from src.utils.config import Settings  # noqa: E402
from tests.fakes import HashEmbedder, QuoteAnswerer  # noqa: E402
from tests.site import acme_site  # noqa: E402

STATE_URL = os.getenv("TEST_DATABASE_URL")
INDEX_URL = os.getenv("TEST_INDEX_DATABASE_URL")
DOCQA = (ROOT / os.getenv("DOCQA_PATH", "../document-qa-citations")).resolve()  # same rule as the app


def _truncate(url: str, tables: str) -> None:
    eng = create_engine(url.replace("postgresql://", "postgresql+psycopg://", 1))
    with eng.begin() as c:
        if c.execute(text("SELECT to_regclass(:t)"), {"t": tables.split(",")[0].strip()}).scalar():
            c.execute(text(f"TRUNCATE {tables} CASCADE"))
    eng.dispose()


@pytest.fixture
def site():
    s = acme_site().start()
    yield s
    s.stop()


@pytest.fixture
def settings(tmp_path):
    if not (STATE_URL and INDEX_URL):
        pytest.skip("TEST_DATABASE_URL / TEST_INDEX_DATABASE_URL not set")
    _truncate(STATE_URL, "sources, pages, page_versions, sync_runs, sync_events, alerts")
    _truncate(INDEX_URL, "documents, chunks")
    return Settings(database_url=STATE_URL, index_database_url=INDEX_URL,
                    docqa_path=DOCQA, chunk_chars=300, chunk_overlap=0,
                    per_host_delay=0.0, allow_private_urls=True, scheduler_enabled=False, log_dir=tmp_path)


def _build(settings):
    return build_app(settings, embedder=HashEmbedder(), answerer=QuoteAnswerer(), checkpointer=None)


@pytest.fixture
def app(settings):
    return _build(settings)


@pytest.fixture
def make_app(settings):
    return lambda **overrides: _build(replace(settings, **overrides))
