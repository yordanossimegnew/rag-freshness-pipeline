"""Staleness metrics and alerting.

A page is *fresh* when it was confirmed (304, unchanged hash or re-indexed) within its
source's SLA. Pages whose last fetch failed keep serving their last good version, so they
age past the SLA and show up here, instead of silently answering with old content.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from ..tools.state import State, now
from ..utils import get_logger

log = get_logger("freshness")

AGE_BUCKETS = [("< 1h", 1), ("1–6h", 6), ("6–24h", 24), ("1–7d", 24 * 7), ("> 7d", None)]


def _dt(v) -> datetime | None:
    return datetime.fromisoformat(v) if isinstance(v, str) else v


def page_freshness(page: dict, sla_hours: int, at: datetime | None = None) -> dict:
    at = at or now()
    checked, changed = _dt(page.get("last_checked_at")), _dt(page.get("last_changed_at"))
    age_h = (at - checked).total_seconds() / 3600 if checked else None
    stale = page["status"] != "gone" and (age_h is None or age_h > sla_hours)
    return {"age_hours": round(age_h, 2) if age_h is not None else None,
            "content_age_hours": round((at - changed).total_seconds() / 3600, 2) if changed else None,
            "stale": stale}


def bucket(age_h: float | None) -> str:
    if age_h is None:
        return "never"
    for label, limit in AGE_BUCKETS:
        if limit is None or age_h < limit:
            return label
    return AGE_BUCKETS[-1][0]


def metrics(state: State, failure_threshold: int, at: datetime | None = None) -> dict:
    at = at or now()
    sources = {s["id"]: s for s in state.list_sources()}
    per_source, histogram = {}, {label: 0 for label, _ in AGE_BUCKETS} | {"never": 0}
    total = fresh = 0
    for sid, src in sources.items():
        pages = [p for p in state.pages_of(sid) if p["status"] != "gone"]
        rows = [{**p, **page_freshness(p, src["sla_hours"], at)} for p in pages]
        for r in rows:
            histogram[bucket(r["age_hours"])] += 1
        n_fresh = sum(1 for r in rows if not r["stale"])
        total += len(rows)
        fresh += n_fresh
        per_source[sid] = {
            "id": sid, "name": src["name"], "pages": len(rows), "fresh": n_fresh, "stale": len(rows) - n_fresh,
            "errors": sum(1 for r in rows if r["status"] in ("error", "blocked")),
            "fresh_pct": round(100 * n_fresh / len(rows), 1) if rows else 100.0,
            "sla_hours": src["sla_hours"], "last_success_at": src["last_success_at"],
            "consecutive_failures": src["consecutive_failures"],
            "failing": src["consecutive_failures"] >= failure_threshold,
            "oldest_hours": max((r["age_hours"] or 0 for r in rows), default=0),
            "next_sync_at": src["next_sync_at"], "syncing": src["syncing"], "enabled": src["enabled"],
        }
    return {"pages": total, "fresh": fresh, "stale": total - fresh,
            "fresh_pct": round(100 * fresh / total, 1) if total else 100.0,
            "age_histogram": histogram, "sources": list(per_source.values())}


def evaluate_alerts(state: State, failure_threshold: int, webhook_url: str | None = None,
                    at: datetime | None = None) -> list[dict]:
    """Open alerts for current problems, resolve the ones that cleared. Returns newly opened alerts."""
    m = metrics(state, failure_threshold, at)
    active: set[str] = set()
    opened = []

    def alert(key, kind, severity, message, sid):
        active.add(key)
        if state.raise_alert(key, kind, severity, message, sid):
            opened.append({"key": key, "kind": kind, "severity": severity, "message": message})

    for src in m["sources"]:
        if not src["enabled"]:
            continue
        if src["failing"]:
            alert(f"source-failing:{src['id']}", "source_failing", "critical",
                  f"{src['name']}: last {src['consecutive_failures']} syncs failed", src["id"])
        if src["stale"]:
            alert(f"stale:{src['id']}", "stale_pages", "critical" if src["fresh_pct"] < 50 else "warning",
                  f"{src['name']}: {src['stale']}/{src['pages']} pages older than the {src['sla_hours']}h SLA",
                  src["id"])
        if src["errors"]:
            alert(f"page-errors:{src['id']}", "page_errors", "warning",
                  f"{src['name']}: {src['errors']} page{'s' if src['errors'] != 1 else ''} failing to fetch "
                  "(serving the last good version)",
                  src["id"])
    state.resolve_alerts(active)
    if webhook_url and opened:
        try:
            httpx.post(webhook_url, json={"text": "\n".join(f"[{a['severity']}] {a['message']}" for a in opened),
                                          "alerts": opened}, timeout=10)
        except httpx.HTTPError as exc:
            log.warning("alert webhook failed: %s", exc)
    return opened


def default_sla(schedule_minutes: int) -> int:
    """Two missed syncs (at least 1h) before a page counts as stale."""
    return max(1, int(timedelta(minutes=2 * schedule_minutes).total_seconds() // 3600))
