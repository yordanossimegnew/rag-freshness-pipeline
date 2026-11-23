"""One sync of one source, as a LangGraph graph.

    START -> discover --Send x page--> sync_page -> reconcile -> finalize -> END

* discover  - expand the source into target URLs (sitemap or URL list, include/exclude filters)
* sync_page - per page, in parallel: skip if the sitemap <lastmod> is unchanged, otherwise a
              conditional GET (ETag / Last-Modified -> 304). On a 200 the extracted text is
              hashed; only a real content change reaches the index, where the chunk-level diff
              re-embeds just the chunks whose text changed. 404/410 removes the page.
* reconcile - pages that disappeared from the sitemap / URL list are removed from the index
* finalize  - run stats, next schedule, failure counter, then alerts are re-evaluated
"""

from __future__ import annotations

import difflib
import hashlib
import operator
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..tools.fetcher import Fetcher, discover_sitemap
from ..tools.index import IndexWriter
from ..tools.state import State, now
from ..utils import get_logger
from ..utils.config import Settings

log = get_logger("sync")


@dataclass
class Components:
    settings: Settings
    state: State
    fetcher: Fetcher
    index: IndexWriter
    checkpointer: Any = None


class SyncState(TypedDict, total=False):
    source: dict
    run_id: str
    targets: list[dict]
    discover_error: str | None
    results: Annotated[list[dict], operator.add]
    stats: dict


def doc_text(doc) -> str:
    parts = [doc.title]
    for s in doc.sections:
        parts.append(f"{s.heading or ''}\n{s.text}".strip())
    return "\n\n".join(p for p in parts if p)


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def text_diff(old: str, new: str, limit: int = 14) -> dict:
    """Line-level summary of what changed, for the change log."""
    old_lines = [l.strip() for l in old.splitlines() if l.strip()]
    new_lines = [l.strip() for l in new.splitlines() if l.strip()]
    added, removed, sample = 0, 0, []
    for line in difflib.unified_diff(old_lines, new_lines, lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
        else:
            continue
        if len(sample) < limit:
            sample.append(line[0] + " " + line[1:][:200])
    return {"lines_added": added, "lines_removed": removed, "sample": sample}


def filter_urls(urls: list[dict], include: str | None, exclude: str | None, max_pages: int) -> list[dict]:
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    kept = [u for u in urls if (not inc or inc.search(u["url"])) and not (exc and exc.search(u["url"]))]
    return kept[:max_pages]


def build_sync_graph(parts: Components):
    s, st, fetcher, index = parts.settings, parts.state, parts.fetcher, parts.index

    # ------------------------------------------------------------------ discover
    def discover(state: SyncState) -> dict:
        src = state["source"]
        try:
            if src["kind"] == "sitemap":
                found = discover_sitemap(fetcher, src["sitemap_url"], max_pages=max(src["max_pages"] * 4, 50))
            else:
                found = [{"url": u, "lastmod": None} for u in src["urls"]]
        except Exception as exc:  # noqa: BLE001 - a broken sitemap fails the run, not the process
            log.warning("discover failed for %s: %s", src["name"], exc)
            return {"targets": [], "discover_error": str(exc)}
        return {"targets": filter_urls(found, src.get("include"), src.get("exclude"), src["max_pages"]),
                "discover_error": None}

    def fan_out(state: SyncState):
        if state.get("discover_error") or not state["targets"]:
            return "reconcile"
        return [Send("sync_page", {"source": state["source"], "run_id": state["run_id"], "target": t})
                for t in state["targets"]]

    # ------------------------------------------------------------------ one page
    def sync_page(payload: dict) -> dict:
        src, rid, target = payload["source"], payload["run_id"], payload["target"]
        url = target["url"]
        page = st.ensure_page(src["id"], url)
        result = _sync_one(src, rid, page, target)
        st.event(rid, url, result["action"], result.get("detail", {}))
        return {"results": [result]}

    def _sync_one(src: dict, rid: str, page: dict, target: dict) -> dict:
        url = target["url"]
        if (target.get("lastmod") and page.get("sitemap_lastmod") == target["lastmod"]
                and page.get("content_hash") and page["status"] == "active"):
            st.update_page(page["id"], last_checked_at=now())
            return {"url": url, "action": "unchanged", "detail": {"reason": "sitemap lastmod unchanged"}}

        fetched = fetcher.fetch(url, page.get("etag"), page.get("last_modified"))
        if fetched.status == "unchanged":
            st.update_page(page["id"], last_checked_at=now(), status="active", last_error=None,
                           sitemap_lastmod=target.get("lastmod"))
            return {"url": url, "action": "unchanged", "detail": {"reason": "304 Not Modified"}, "http_304": True}
        if fetched.status == "gone":
            removed = index.remove(url)
            st.update_page(page["id"], status="gone", document_id=None, last_checked_at=now(),
                           last_changed_at=now() if removed else page.get("last_changed_at"),
                           last_error=f"HTTP {fetched.http_status}", content_hash=None, etag=None,
                           last_modified=None)
            return {"url": url, "action": "removed" if removed else "skipped",
                    "detail": {"reason": f"HTTP {fetched.http_status}"}}
        if fetched.status in ("error", "blocked"):
            # keep serving the last good version; the page just gets staler (and alerts fire)
            st.update_page(page["id"], status="error" if fetched.status == "error" else "blocked",
                           last_error=fetched.error)
            return {"url": url, "action": "error" if fetched.status == "error" else "skipped",
                    "detail": {"error": fetched.error}}

        try:
            doc = index.parse(url, fetched.body, fetched.content_type)
        except Exception as exc:  # noqa: BLE001 - unparseable page: record, keep old version
            st.update_page(page["id"], status="error", last_error=f"parse: {exc}")
            return {"url": url, "action": "error", "detail": {"error": f"parse: {exc}"}}

        text = doc_text(doc)
        digest = content_hash(text)
        meta = dict(etag=fetched.etag, last_modified=fetched.last_modified, sitemap_lastmod=target.get("lastmod"),
                    last_checked_at=now(), status="active", last_error=None, title=doc.title)
        if digest == page.get("content_hash") and page.get("document_id"):
            st.update_page(page["id"], **meta)  # 200 but same text (e.g. a timestamp in a header we strip)
            return {"url": url, "action": "unchanged", "detail": {"reason": "content hash unchanged"},
                    "bytes": len(fetched.body)}

        previous = st.versions(page["id"])
        up = index.upsert(url, doc)
        diff = text_diff(previous[0]["text"], text) if previous else None
        stats = {"chunks": up.chunks, "reused": up.reused, "embedded": up.embedded, "removed": up.removed}
        st.add_version(page["id"], digest, doc.title, text, stats | ({"diff": diff} if diff else {}), s.keep_versions)
        st.update_page(page["id"], document_id=up.document_id, content_hash=digest, last_changed_at=now(), **meta)
        action = "added" if (up.created or not page.get("content_hash")) else "updated"
        return {"url": url, "action": action, "bytes": len(fetched.body), "embed_tokens": up.embed_tokens,
                "detail": {"title": doc.title, **stats, **({"diff": diff} if diff else {})}}

    # ------------------------------------------------------------------ reconcile + finalize
    def reconcile(state: SyncState) -> dict:
        if state.get("discover_error"):
            return {}
        wanted = {t["url"] for t in state["targets"]}
        out = []
        for page in st.pages_of(state["source"]["id"]):
            if page["url"] in wanted or page["status"] == "gone":
                continue
            removed = index.remove(page["url"])
            st.update_page(page["id"], status="gone", document_id=None, content_hash=None, last_checked_at=now(),
                           last_changed_at=now(), last_error="no longer listed in the source")
            detail = {"reason": "no longer in the sitemap" if state["source"]["kind"] == "sitemap"
                      else "removed from the source's URL list"}
            st.event(state["run_id"], page["url"], "removed" if removed else "skipped", detail)
            out.append({"url": page["url"], "action": "removed" if removed else "skipped", "detail": detail})
        return {"results": out}

    def finalize(state: SyncState) -> dict:
        results = state.get("results") or []
        count = lambda a: sum(1 for r in results if r["action"] == a)  # noqa: E731
        stats = {
            "pages": len(state.get("targets") or []),
            "added": count("added"), "updated": count("updated"), "unchanged": count("unchanged"),
            "removed": count("removed"), "errors": count("error"), "skipped": count("skipped"),
            "not_modified_304": sum(1 for r in results if r.get("http_304")),
            "chunks_embedded": sum(r.get("detail", {}).get("embedded", 0) for r in results),
            "chunks_reused": sum(r.get("detail", {}).get("reused", 0) for r in results),
            "chunks_removed": sum(r.get("detail", {}).get("removed", 0) for r in results),
            "embed_tokens": sum(r.get("embed_tokens", 0) for r in results),
            "bytes_downloaded": sum(r.get("bytes", 0) for r in results),
        }
        stats["embed_cost_usd"] = round(stats["embed_tokens"] * s.voyage_embed_price / 1e6, 6)
        failed = bool(state.get("discover_error")) or (stats["pages"] > 0 and stats["errors"] == stats["pages"])
        return {"stats": stats | {"failed": failed}}

    g = StateGraph(SyncState)
    g.add_node("discover", discover)
    g.add_node("sync_page", sync_page)
    g.add_node("reconcile", reconcile)
    g.add_node("finalize", finalize)
    g.add_edge(START, "discover")
    g.add_conditional_edges("discover", fan_out, ["sync_page", "reconcile"])
    g.add_edge("sync_page", "reconcile")
    g.add_edge("reconcile", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=parts.checkpointer)


def run_sync(parts: Components, source_id: str, trigger: str = "manual", graph=None) -> dict:
    """Run one sync (callers must have claimed the source). Always releases the claim."""
    st, s = parts.state, parts.settings
    source = st.source(source_id)
    rid = st.start_run(source_id, trigger)
    try:
        graph = graph or build_sync_graph(parts)
        out = graph.invoke({"source": source, "run_id": rid, "results": []},
                           {"configurable": {"thread_id": f"sync-{rid}"}, "max_concurrency": s.max_concurrency})
        stats = out["stats"]
        failed = stats.pop("failed")
        st.finish_run(rid, "failed" if failed else "completed", stats, out.get("discover_error"))
        st.update_source(source_id, syncing=False, last_sync_at=now(),
                         next_sync_at=now() + timedelta(minutes=source["schedule_minutes"]),
                         consecutive_failures=source["consecutive_failures"] + 1 if failed else 0,
                         **({} if failed else {"last_success_at": now()}))
    except Exception as exc:
        log.exception("sync %s crashed", rid)
        st.finish_run(rid, "failed", {}, f"{type(exc).__name__}: {exc}")
        st.update_source(source_id, syncing=False, last_sync_at=now(),
                         next_sync_at=now() + timedelta(minutes=source["schedule_minutes"]),
                         consecutive_failures=source["consecutive_failures"] + 1)
    return st.run(rid)
