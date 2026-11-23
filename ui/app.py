"""Streamlit UI for the RAG Freshness Pipeline (talks to the FastAPI app)."""

from __future__ import annotations

import html
import os
from datetime import datetime, timezone
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.getenv("AGENT_API_URL", "http://127.0.0.1:8000")
BLUE, ORANGE = "#2a78d6", "#eb6834"

st.set_page_config(page_title="RAG Freshness", page_icon="🔄", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem;}
.diff {font-family: ui-monospace, Consolas, monospace; font-size: 0.82rem; border: 1px solid #e6e6e6;
       border-radius: 6px; padding: 6px 10px; background: #fafafa; white-space: pre-wrap;}
.diff .add {background: #e8f6ec; color: #116329; display: block;}
.diff .del {background: #fdecec; color: #a01c1c; display: block;}
.chip {display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 0.78rem; margin-right: 4px;
       border: 1px solid #ddd; color: #444; background: #f6f6f6;}
.chip.fresh {border-color: #bfe3c9; background: #eef8f1; color: #116329;}
.chip.stale {border-color: #f3c7a8; background: #fdf1e8; color: #9a4210;}
.answer {font-size: 1.05rem; line-height: 1.6; padding: 12px 16px; border-left: 3px solid #2a78d6;
         background: #f7faff; border-radius: 4px;}
</style>""", unsafe_allow_html=True)


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kw)
    except requests.RequestException as exc:
        st.error(f"API not reachable at {API}: {exc}")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        return {"error": detail, "status": r.status_code}
    return r.json() if r.content else {}


def ago(iso: str | None) -> str:
    if not iso:
        return "never"
    secs = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{secs / 3600:.1f} h ago"
    return f"{secs / 86400:.1f} d ago"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def until(iso: str | None) -> str:
    if not iso:
        return "—"
    secs = (datetime.fromisoformat(iso) - datetime.now(timezone.utc)).total_seconds()
    if secs <= 0:
        return "due now"
    return f"in {int(secs // 60)} min" if secs < 3600 else f"in {secs / 3600:.1f} h"


def short(url: str) -> str:
    return urlparse(url).path or url


# ----------------------------------------------------------------------------- sidebar
health = api("GET", "/health")
with st.sidebar:
    st.title("🔄 RAG Freshness")
    st.caption("Keeps the Document QA index in sync with the web")
    st.markdown(f"**Answers:** `{health.get('answer_model', '?')}`")
    idx = health.get("index", {})
    st.markdown(f"**Index:** {idx.get('documents', 0)} documents · {idx.get('chunks', 0)} chunks")
    st.markdown("**Scheduler:** " + ("on" if health.get("scheduler") else "off"))

tab_fresh, tab_log, tab_ask, tab_sources = st.tabs(["📈 Freshness", "📝 Change log", "💬 Ask", "🌐 Sources"])

# ----------------------------------------------------------------------------- freshness
with tab_fresh:
    m = api("GET", "/metrics")
    alerts = api("GET", "/alerts")
    k = st.columns(4)
    k[0].metric("Fresh pages", f"{m['fresh_pct']:.0f}%")
    k[0].caption(f"{m['fresh']} of {m['pages']} confirmed within their SLA")
    k[1].metric("Stale pages", m["stale"])
    k[1].caption("older than their source's SLA")
    k[2].metric("Sources", len(m["sources"]))
    k[2].caption(f"{sum(s['failing'] for s in m['sources'])} failing")
    k[3].metric("Open alerts", len(alerts))
    k[3].caption(f"{sum(a['severity'] == 'critical' for a in alerts)} critical")

    if alerts:
        st.subheader("Alerts")
        for a in alerts:
            icon = "🔴" if a["severity"] == "critical" else "🟠"
            c1, c2 = st.columns([8, 1])
            c1.markdown(f"{icon} **{a['kind'].replace('_', ' ')}** · {a['message']} · opened {ago(a['created_at'])}")
            if c2.button("Resolve", key=f"res{a['id']}"):
                api("POST", f"/alerts/{a['id']}/resolve")
                st.rerun()

    st.subheader("Sources")
    rows = [{"source": s["name"], "fresh": s["fresh_pct"] / 100, "pages": s["pages"], "stale": s["stale"],
             "fetch errors": s["errors"], "SLA": f"{s['sla_hours']}h", "last success": ago(s["last_success_at"]),
             "failed syncs in a row": s["consecutive_failures"],
             "next sync": "running…" if s["syncing"] else until(s["next_sync_at"])} for s in m["sources"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            "fresh": st.column_config.ProgressColumn("fresh", format="percent", min_value=0, max_value=1)})
    else:
        st.info("No sources yet: add one in the Sources tab.")

    left, _ = st.columns([1, 1])
    with left:
        st.subheader("Time since last confirmed")
        order = ["< 1h", "1–6h", "6–24h", "1–7d", "> 7d", "never"]
        hist = pd.DataFrame({"age": order, "pages": [m["age_histogram"].get(b, 0) for b in order]})
        chart = alt.Chart(hist).mark_bar(color=BLUE, cornerRadiusEnd=4, size=26).encode(
            x=alt.X("age:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pages:Q", title="pages", axis=alt.Axis(tickMinStep=1, grid=True, gridColor="#eee")),
            tooltip=["age", "pages"]).properties(height=230).configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)

# ----------------------------------------------------------------------------- change log
with tab_log:
    runs = api("GET", "/runs", params={"limit": 30})
    names = {s["id"]: s["name"] for s in api("GET", "/sources")}
    if not runs:
        st.info("No syncs yet.")
    for i, r in enumerate(runs):
        s = r.get("stats") or {}
        badge = {"completed": "✅", "failed": "❌", "running": "⏳"}[r["status"]]
        title = (f"{badge} {names.get(r['source_id'], '?')} · {r['trigger']} · {ago(r['started_at'])} — "
                 f"+{s.get('added', 0)} added · ~{s.get('updated', 0)} updated · ={s.get('unchanged', 0)} unchanged"
                 f" · −{s.get('removed', 0)} removed · {plural(s.get('errors', 0), 'error')}")
        with st.expander(title, expanded=i == 0):
            if r["status"] == "running":
                st.info("Sync in progress…")
                continue
            if r.get("error"):
                st.error(r["error"])
            c = st.columns(4)
            c[0].metric("Chunks embedded", s.get("chunks_embedded", 0))
            c[1].metric("Chunks reused", s.get("chunks_reused", 0))
            c[2].metric("304 Not Modified", s.get("not_modified_304", 0))
            c[3].metric("Downloaded", f"{s.get('bytes_downloaded', 0) / 1024:.1f} KB")
            detail = api("GET", f"/runs/{r['id']}")
            quiet = [e for e in detail["events"] if e["action"] == "unchanged"]
            for e in detail["events"]:
                if e["action"] == "unchanged":
                    continue
                d = e["detail"]
                icon = {"added": "🟢", "updated": "🔵", "removed": "⚫", "error": "🔴", "skipped": "⚪"}[e["action"]]
                extra = ""
                if e["action"] in ("added", "updated"):
                    extra = f" · {plural(d.get('chunks', 0), 'chunk')}: {d.get('embedded', 0)} re-embedded, {d.get('reused', 0)} reused"
                elif d.get("reason") or d.get("error"):
                    extra = f" · {d.get('reason') or d.get('error')}"
                st.markdown(f"{icon} **{e['action']}** `{short(e['url'])}`{extra}")
                diff = d.get("diff")
                if diff and diff["sample"]:
                    lines = "".join(f"<span class='{'add' if l[0] == '+' else 'del'}'>{html.escape(l)}</span>"
                                    for l in diff["sample"])
                    st.markdown(f"<div class='diff'>{lines}</div>", unsafe_allow_html=True)
            if quiet:
                st.caption(f"{len(quiet)} unchanged: " + ", ".join(sorted({e['detail'].get('reason', '') for e in quiet})))

# ----------------------------------------------------------------------------- ask
with tab_ask:
    with st.form("ask"):
        q = st.text_input("Question", placeholder="How much does the Pro plan cost?")
        go = st.form_submit_button("Ask", type="primary")
    if go and q.strip():
        with st.spinner("Retrieving fresh chunks and answering…"):
            st.session_state["answer"] = api("POST", "/ask", json={"question": q})
    ans = st.session_state.get("answer")
    if ans and "error" in ans:
        st.error(ans["error"])
    elif ans:
        if ans.get("any_stale"):
            st.warning("Some sources are past their freshness SLA; the answer may be out of date.")
        st.markdown(f"<div class='answer'>{html.escape(ans['markdown'])}</div>", unsafe_allow_html=True)
        st.write("")
        for src in ans["sources"]:
            chip = "<span class='chip stale'>stale</span>" if src["stale"] else "<span class='chip fresh'>fresh</span>"
            st.markdown(f"**[{src['n']}] {html.escape(src['label'])}** {chip}"
                        f"<span class='chip'>as of {ago(src['as_of'])}</span>"
                        f"<span class='chip'>content changed {ago(src['changed_at'])}</span>",
                        unsafe_allow_html=True)
            for quote in src["quotes"]:
                st.caption(("✓ " if quote["verified"] else "✗ ") + f"“{quote['text']}”")
            st.caption(src["source"])

# ----------------------------------------------------------------------------- sources
with tab_sources:
    for s in api("GET", "/sources"):
        f = s.get("freshness") or {}
        with st.container(border=True):
            c1, c2 = st.columns([6, 1])
            what = s["sitemap_url"] or f"{len(s['urls'])} URLs"
            c1.markdown(f"**{s['name']}** · `{what}` · every {s['schedule_minutes']} min · SLA {s['sla_hours']}h"
                        f" · {f.get('pages', 0)} pages, {f.get('fresh', 0)} fresh")
            if c2.button("Sync now", key=f"sync{s['id']}", disabled=s["syncing"], type="primary"):
                res = api("POST", f"/sources/{s['id']}/sync")
                if "error" in res:
                    st.warning(res["error"])
                else:
                    st.toast(f"Sync of {s['name']} started")
            pages = api("GET", "/pages", params={"source_id": s["id"]})
            if pages:
                st.dataframe(pd.DataFrame([{
                    "page": short(p["url"]), "status": p["status"], "fresh": "stale" if p["stale"] else "fresh",
                    "checked": ago(p["last_checked_at"]), "content changed": ago(p["last_changed_at"]),
                    "error": p["last_error"] or ""} for p in pages]), hide_index=True, use_container_width=True)
                pick = st.selectbox("Page history", [None] + [p["id"] for p in pages if p["last_changed_at"]],
                                    format_func=lambda pid: "—" if pid is None else
                                    short(next(p["url"] for p in pages if p["id"] == pid)), key=f"hist{s['id']}")
                if pick:
                    for v in api("GET", f"/pages/{pick}")["versions"]:
                        st.markdown(f"**{v['fetched_at'][:19].replace('T', ' ')}** · {v['stats'].get('chunks', 0)} chunks"
                                    f" ({v['stats'].get('embedded', 0)} embedded)")
                        if v["stats"].get("diff"):
                            lines = "".join(f"<span class='{'add' if l[0] == '+' else 'del'}'>{html.escape(l)}</span>"
                                            for l in v["stats"]["diff"]["sample"])
                            st.markdown(f"<div class='diff'>{lines}</div>", unsafe_allow_html=True)

    with st.expander("➕ Add a source", expanded=False):
        with st.form("add"):
            name = st.text_input("Name")
            kind = st.radio("Kind", ["Sitemap", "URL list"], horizontal=True)
            target = st.text_area("Sitemap URL, or one URL per line")
            c = st.columns(3)
            every = c[0].number_input("Sync every (minutes)", 1, 60 * 24 * 7, 60)
            sla = c[1].number_input("SLA (hours, 0 = auto)", 0, 24 * 30, 0)
            include = c[2].text_input("Only URLs matching (regex)")
            if st.form_submit_button("Add source"):
                body = {"name": name, "schedule_minutes": every, "include": include or None,
                        "sla_hours": sla or None}
                lines = [l.strip() for l in target.splitlines() if l.strip()]
                body |= {"sitemap_url": lines[0] if lines else None} if kind == "Sitemap" else {"urls": lines}
                res = api("POST", "/sources", json=body)
                st.error(res["error"]) if "error" in res else st.rerun()
