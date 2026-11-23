"""FastAPI application: sources, syncs, change log, page history, freshness, alerts, time-aware Q&A."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl, model_validator

from ..agent import App, Scheduler, build_app, build_qa_graph, metrics, page_freshness, run_claimed
from ..agent.freshness import default_sla
from ..utils import get_logger, setup_logging
from ..utils.config import Settings, load_settings

log = get_logger("api")


class SourceIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    sitemap_url: HttpUrl | None = None
    urls: list[HttpUrl] = Field(default_factory=list, max_length=2000)
    include: str | None = Field(None, description="Regex: only sync URLs matching it.")
    exclude: str | None = Field(None, description="Regex: skip URLs matching it.")
    schedule_minutes: int = Field(60, ge=1, le=60 * 24 * 30)
    sla_hours: int | None = Field(None, ge=1, description="Default: two missed syncs (min 1h).")
    max_pages: int = Field(200, ge=1, le=5000)

    @model_validator(mode="after")
    def one_kind(self):
        if bool(self.sitemap_url) == bool(self.urls):
            raise ValueError("give either sitemap_url or urls")
        return self


class SourcePatch(BaseModel):
    enabled: bool | None = None
    schedule_minutes: int | None = Field(None, ge=1)
    sla_hours: int | None = Field(None, ge=1)
    include: str | None = None
    exclude: str | None = None


class AskIn(BaseModel):
    question: str = Field(min_length=2, max_length=4000)


def create_app(settings: Settings | None = None, app_parts: App | None = None, run_inline: bool = False,
               scheduler: bool | None = None) -> FastAPI:
    settings = settings or load_settings()
    setup_logging(settings.log_level, settings.log_dir)
    holder: dict = {"app": app_parts}
    lock = threading.Lock()

    def parts() -> App:
        with lock:
            if holder["app"] is None:
                holder["app"] = build_app(settings)
            return holder["app"]

    @asynccontextmanager
    async def lifespan(api: FastAPI):
        p = parts()
        p.parts.state.release_all()
        sched = None
        if settings.scheduler_enabled if scheduler is None else scheduler:
            sched = Scheduler(p.parts)
            sched.start()
        yield
        if sched:
            sched.stop()

    api = FastAPI(title="RAG Freshness Pipeline", version="1.0.0", lifespan=lifespan,
                  description="Keeps Document QA's pgvector index in sync with web pages and sitemaps.")

    def _source(sid: str) -> dict:
        src = parts().parts.state.source(sid)
        if not src:
            raise HTTPException(404, "source not found")
        return src

    @api.get("/health")
    def health() -> dict:
        p = parts()
        return {"status": "ok", "answer_model": getattr(p._answerer, "name", settings.model),
                "index": p.parts.index.store.counts(), "scheduler": settings.scheduler_enabled}

    # ------------------------------------------------------------------ sources
    @api.get("/sources")
    def list_sources() -> list[dict]:
        m = {s["id"]: s for s in metrics(parts().parts.state, settings.failure_alert_threshold)["sources"]}
        return [{**s, "freshness": m.get(s["id"])} for s in parts().parts.state.list_sources()]

    @api.post("/sources", status_code=201)
    def add_source(body: SourceIn) -> dict:
        check = parts().parts.fetcher.check_url
        for u in ([body.sitemap_url] if body.sitemap_url else body.urls):
            try:
                check(str(u), settings.allow_private_urls)
            except ValueError as exc:
                raise HTTPException(422, f"{u}: {exc}") from exc
        return parts().parts.state.add_source(
            name=body.name, kind="sitemap" if body.sitemap_url else "urls",
            sitemap_url=str(body.sitemap_url) if body.sitemap_url else None, urls=[str(u) for u in body.urls],
            include=body.include, exclude=body.exclude, schedule_minutes=body.schedule_minutes,
            sla_hours=body.sla_hours or default_sla(body.schedule_minutes), max_pages=body.max_pages)

    @api.get("/sources/{sid}")
    def get_source(sid: str) -> dict:
        return _source(sid) | {"pages": pages(sid)}

    @api.patch("/sources/{sid}")
    def patch_source(sid: str, body: SourcePatch) -> dict:
        _source(sid)
        parts().parts.state.update_source(sid, **body.model_dump(exclude_none=True))
        return _source(sid)

    @api.delete("/sources/{sid}", status_code=204)
    def delete_source(sid: str, purge: bool = Query(True, description="Also remove its pages from the index.")):
        p = parts().parts
        _source(sid)
        if purge:
            for page in p.state.pages_of(sid):
                p.index.remove(page["url"])
        p.state.delete_source(sid)

    @api.post("/sources/{sid}/sync", status_code=202)
    def sync(sid: str) -> dict:
        _source(sid)
        p = parts().parts
        if not p.state.claim_source(sid):
            raise HTTPException(409, "a sync of this source is already running")
        if run_inline:
            return run_claimed(p, sid, "manual")
        threading.Thread(target=run_claimed, args=(p, sid, "manual"), daemon=True).start()
        return {"status": "started", "source_id": sid}

    # ------------------------------------------------------------------ runs and pages
    @api.get("/runs")
    def runs(source_id: str | None = None, limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        return parts().parts.state.runs(source_id, limit)

    @api.get("/runs/{rid}")
    def run(rid: str) -> dict:
        r = parts().parts.state.run(rid)
        if not r:
            raise HTTPException(404, "run not found")
        return r

    @api.get("/pages")
    def pages(source_id: str | None = None) -> list[dict]:
        st = parts().parts.state
        slas = {s["id"]: s["sla_hours"] for s in st.list_sources()}
        return [{**p, **page_freshness(p, slas.get(p["source_id"], 24))} for p in st.pages_of(source_id)]

    @api.get("/pages/{pid}")
    def page(pid: int) -> dict:
        st = parts().parts.state
        p = st.page_by_id(pid)
        if not p:
            raise HTTPException(404, "page not found")
        src = st.source(p["source_id"])
        return {**p, **page_freshness(p, src["sla_hours"]), "versions": st.versions(pid)}

    # ------------------------------------------------------------------ freshness
    @api.get("/metrics")
    def get_metrics() -> dict:
        return metrics(parts().parts.state, settings.failure_alert_threshold)

    @api.get("/alerts")
    def alerts() -> list[dict]:
        return parts().parts.state.open_alerts()

    @api.post("/alerts/{aid}/resolve")
    def resolve(aid: int) -> dict:
        if not parts().parts.state.resolve_alert(aid):
            raise HTTPException(404, "no open alert with that id")
        return {"resolved": aid}

    # ------------------------------------------------------------------ Q&A
    @api.post("/ask")
    def ask(body: AskIn) -> dict:
        p = parts()
        try:
            graph = build_qa_graph(settings, p.parts.state, p.parts.index, p.embedder, p.answerer)
            return graph.invoke({"question": body.question})["result"]
        except Exception as exc:
            log.exception("ask failed")
            raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc

    return api

