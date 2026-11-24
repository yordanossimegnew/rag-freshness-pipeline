"""Pipeline state in PostgreSQL: sources, pages, page versions, sync runs, change events, alerts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

metadata = MetaData()

sources = Table(
    "sources", metadata,
    Column("id", String(32), primary_key=True),
    Column("name", Text, nullable=False),
    Column("kind", String(10), nullable=False),            # urls | sitemap
    Column("urls", JSONB, nullable=False, default=list),
    Column("sitemap_url", Text),
    Column("include", Text),                               # regex: only URLs matching
    Column("exclude", Text),                               # regex: skip URLs matching
    Column("schedule_minutes", Integer, nullable=False),
    Column("sla_hours", Integer, nullable=False),          # a page unchecked for longer is stale
    Column("max_pages", Integer, nullable=False),
    Column("enabled", Boolean, nullable=False, default=True),
    Column("next_sync_at", DateTime(timezone=True), nullable=False),
    Column("last_sync_at", DateTime(timezone=True)),
    Column("last_success_at", DateTime(timezone=True)),
    Column("consecutive_failures", Integer, nullable=False, default=0),
    Column("syncing", Boolean, nullable=False, default=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
pages = Table(
    "pages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_id", String(32), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("url", Text, nullable=False),
    Column("document_id", String(32)),                     # id in Document QA's index
    Column("title", Text),
    Column("status", String(10), nullable=False),          # active | gone | error | blocked
    Column("etag", Text),
    Column("last_modified", Text),
    Column("sitemap_lastmod", Text),
    Column("content_hash", String(64)),
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
    Column("last_checked_at", DateTime(timezone=True)),    # last time we confirmed the page (304 or 200)
    Column("last_changed_at", DateTime(timezone=True)),    # last time its content actually changed
    Column("last_error", Text),
    UniqueConstraint("source_id", "url", name="uq_page_source_url"),
)
page_versions = Table(
    "page_versions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("page_id", Integer, ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("content_hash", String(64), nullable=False),
    Column("title", Text, nullable=False),
    Column("text", Text, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("stats", JSONB, nullable=False, default=dict),
)
sync_runs = Table(
    "sync_runs", metadata,
    Column("id", String(32), primary_key=True),
    Column("source_id", String(32), ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("trigger", String(10), nullable=False),         # schedule | manual
    Column("status", String(10), nullable=False),          # running | completed | failed
    Column("stats", JSONB, nullable=False, default=dict),
    Column("error", Text),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
)
sync_events = Table(
    "sync_events", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(32), ForeignKey("sync_runs.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("url", Text, nullable=False),
    Column("action", String(10), nullable=False),          # added | updated | unchanged | removed | error | skipped
    Column("detail", JSONB, nullable=False, default=dict),
    Column("at", DateTime(timezone=True), nullable=False),
)
alerts = Table(
    "alerts", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("key", Text, nullable=False, index=True),       # dedupe key, e.g. "source-failing:<id>"
    Column("kind", String(20), nullable=False),            # source_failing | stale_pages | page_error
    Column("severity", String(10), nullable=False),        # warning | critical
    Column("message", Text, nullable=False),
    Column("source_id", String(32)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("resolved_at", DateTime(timezone=True)),
)


def now() -> datetime:
    return datetime.now(timezone.utc)


def _row(row) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row._mapping.items()}


class State:
    def __init__(self, database_url: str):
        url = database_url
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                url = "postgresql+psycopg://" + url[len(prefix):]
        self.engine: Engine = create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)

    def init(self) -> None:
        metadata.create_all(self.engine)

    # ------------------------------------------------------------------ sources
    def add_source(self, **values) -> dict:
        sid = uuid.uuid4().hex[:12]
        with self.engine.begin() as c:
            c.execute(insert(sources).values(id=sid, enabled=True, next_sync_at=now(), consecutive_failures=0,
                                             syncing=False, created_at=now(), **values))
        return self.source(sid)

    def source(self, sid: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(sources).where(sources.c.id == sid)).first()
        return _row(row) if row else None

    def list_sources(self) -> list[dict]:
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(select(sources).order_by(sources.c.created_at))]

    def update_source(self, sid: str, **values) -> None:
        with self.engine.begin() as c:
            c.execute(update(sources).where(sources.c.id == sid).values(**values))

    def delete_source(self, sid: str) -> bool:
        with self.engine.begin() as c:
            return bool(c.execute(delete(sources).where(sources.c.id == sid)).rowcount)

    def claim_source(self, sid: str) -> bool:
        """Mark a source as syncing; False if a sync is already running (no overlapping syncs)."""
        with self.engine.begin() as c:
            return bool(c.execute(update(sources).where(sources.c.id == sid, sources.c.syncing.is_(False))
                                  .values(syncing=True)).rowcount)

    def release_all(self) -> None:
        """On startup: a sync that was running when the process died is not running any more."""
        with self.engine.begin() as c:
            c.execute(update(sources).where(sources.c.syncing.is_(True)).values(syncing=False))
        with self.engine.begin() as c:
            c.execute(update(sync_runs).where(sync_runs.c.status == "running")
                      .values(status="failed", error="interrupted (process restarted)", finished_at=now()))

    def pages_by_document(self, document_ids: list[str]) -> dict[str, dict]:
        if not document_ids:
            return {}
        q = (select(pages, sources.c.sla_hours, sources.c.name.label("source_name"))
             .join(sources, sources.c.id == pages.c.source_id).where(pages.c.document_id.in_(document_ids)))
        with self.engine.connect() as c:
            return {r.document_id: _row(r) for r in c.execute(q)}

    def due_sources(self) -> list[dict]:
        with self.engine.connect() as c:
            rows = c.execute(select(sources).where(sources.c.enabled.is_(True), sources.c.syncing.is_(False),
                                                   sources.c.next_sync_at <= now())).all()
        return [_row(r) for r in rows]

    # ------------------------------------------------------------------ pages
    def page(self, source_id: str, url: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(pages).where(pages.c.source_id == source_id, pages.c.url == url)).first()
        return _row(row) if row else None

    def page_by_id(self, pid: int) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(pages).where(pages.c.id == pid)).first()
        return _row(row) if row else None

    def ensure_page(self, source_id: str, url: str) -> dict:
        existing = self.page(source_id, url)
        if existing:
            return existing
        with self.engine.begin() as c:
            c.execute(insert(pages).values(source_id=source_id, url=url, status="active", first_seen_at=now()))
        return self.page(source_id, url)

    def update_page(self, pid: int, **values) -> None:
        with self.engine.begin() as c:
            c.execute(update(pages).where(pages.c.id == pid).values(**values))

    def pages_of(self, source_id: str | None = None) -> list[dict]:
        q = select(pages).order_by(pages.c.url)
        if source_id:
            q = q.where(pages.c.source_id == source_id)
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(q)]

    def add_version(self, page_id: int, content_hash: str, title: str, text: str, stats: dict, keep: int) -> None:
        with self.engine.begin() as c:
            c.execute(insert(page_versions).values(page_id=page_id, content_hash=content_hash, title=title, text=text,
                                                   fetched_at=now(), stats=stats))
            old = [r.id for r in c.execute(select(page_versions.c.id).where(page_versions.c.page_id == page_id)
                                           .order_by(page_versions.c.id.desc()).offset(keep))]
            if old:
                c.execute(delete(page_versions).where(page_versions.c.id.in_(old)))

    def versions(self, page_id: int) -> list[dict]:
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(select(page_versions).where(page_versions.c.page_id == page_id)
                                               .order_by(page_versions.c.id.desc()))]

    # ------------------------------------------------------------------ runs and events
    def start_run(self, source_id: str, trigger: str) -> str:
        rid = uuid.uuid4().hex[:12]
        with self.engine.begin() as c:
            c.execute(insert(sync_runs).values(id=rid, source_id=source_id, trigger=trigger, status="running",
                                               stats={}, started_at=now()))
        return rid

    def finish_run(self, rid: str, status: str, stats: dict, error: str | None = None) -> None:
        with self.engine.begin() as c:
            c.execute(update(sync_runs).where(sync_runs.c.id == rid).values(status=status, stats=stats, error=error,
                                                                             finished_at=now()))

    def event(self, rid: str, url: str, action: str, detail: dict) -> None:
        with self.engine.begin() as c:
            c.execute(insert(sync_events).values(run_id=rid, url=url, action=action, detail=detail, at=now()))

    def runs(self, source_id: str | None = None, limit: int = 50) -> list[dict]:
        q = select(sync_runs).order_by(sync_runs.c.started_at.desc()).limit(limit)
        if source_id:
            q = q.where(sync_runs.c.source_id == source_id)
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(q)]

    def run(self, rid: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(sync_runs).where(sync_runs.c.id == rid)).first()
            if row is None:
                return None
            events = [_row(e) for e in c.execute(select(sync_events).where(sync_events.c.run_id == rid)
                                                  .order_by(sync_events.c.id))]
        return {**_row(row), "events": events}

    # ------------------------------------------------------------------ alerts
    def open_alerts(self) -> list[dict]:
        with self.engine.connect() as c:
            return [_row(r) for r in c.execute(select(alerts).where(alerts.c.resolved_at.is_(None))
                                               .order_by(alerts.c.created_at.desc()))]

    def raise_alert(self, key: str, kind: str, severity: str, message: str, source_id: str | None) -> bool:
        """Open an alert unless one with the same key is already open. Returns True if new."""
        with self.engine.begin() as c:
            exists = c.execute(select(alerts.c.id).where(alerts.c.key == key, alerts.c.resolved_at.is_(None))).first()
            if exists:
                c.execute(update(alerts).where(alerts.c.id == exists.id).values(message=message, severity=severity))
                return False
            c.execute(insert(alerts).values(key=key, kind=kind, severity=severity, message=message,
                                            source_id=source_id, created_at=now()))
            return True

    def resolve_alerts(self, keep_keys: set[str]) -> int:
        """Resolve every open alert whose condition is no longer true."""
        with self.engine.begin() as c:
            q = update(alerts).where(alerts.c.resolved_at.is_(None))
            if keep_keys:
                q = q.where(alerts.c.key.not_in(keep_keys))
            return c.execute(q.values(resolved_at=now())).rowcount

    def resolve_alert(self, alert_id: int) -> bool:
        with self.engine.begin() as c:
            return bool(c.execute(update(alerts).where(alerts.c.id == alert_id, alerts.c.resolved_at.is_(None))
                                  .values(resolved_at=now())).rowcount)
