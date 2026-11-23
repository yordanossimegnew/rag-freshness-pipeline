"""End-to-end syncs against a real local website, Postgres state and Document QA's real index."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select

from src.agent import build_qa_graph, evaluate_alerts, metrics, sync_now
from src.agent.sync import filter_urls, text_diff
from src.tools.fetcher import parse_sitemap
from src.tools.state import now
from tests.site import PRICING_V1, PRICING_V2


def add_sitemap_source(app, site, **kw):
    values = dict(name="Acme docs", kind="sitemap", sitemap_url=site.base + "/sitemap.xml", urls=[],
                  schedule_minutes=60, sla_hours=2, max_pages=50) | kw
    return app.parts.state.add_source(**values)


def events(run, action=None):
    return [e for e in run["events"] if action is None or e["action"] == action]


def index_texts(app):
    ch = app.parts.index.chunks
    with app.parts.index.store.engine.connect() as c:
        return " ".join(r.text for r in c.execute(select(ch.c.text)))


# --------------------------------------------------------------------------- helpers
def test_sitemap_parsing_and_filters():
    xml = (b'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           b'<url><loc>/a</loc><lastmod>2026-11-01</lastmod></url><url><loc>https://x.com/b</loc></url></urlset>')
    children, pages = parse_sitemap(xml, "https://x.com/sitemap.xml")
    assert children == [] and pages == [{"url": "https://x.com/a", "lastmod": "2026-11-01"},
                                        {"url": "https://x.com/b", "lastmod": None}]
    index = b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><sitemap><loc>https://x.com/s1.xml</loc></sitemap></sitemapindex>'
    assert parse_sitemap(index, "https://x.com/")[0] == ["https://x.com/s1.xml"]
    urls = [{"url": u} for u in ["https://x.com/docs/a", "https://x.com/blog/b", "https://x.com/docs/old/c"]]
    assert [u["url"] for u in filter_urls(urls, r"/docs/", r"/old/", 10)] == ["https://x.com/docs/a"]


def test_text_diff_summarises_changed_lines():
    d = text_diff("a\nprice $20\nc", "a\nprice $25\nc\nnew line")
    assert d["lines_added"] == 2 and d["lines_removed"] == 1
    assert d["sample"][:2] == ["- price $20", "+ price $25"]


# --------------------------------------------------------------------------- syncing
def test_first_sync_indexes_pages_and_respects_robots(app, site):
    src = add_sitemap_source(app, site)
    run = sync_now(app.parts, src["id"])
    assert run["status"] == "completed"
    assert run["stats"]["added"] == 4 and run["stats"]["skipped"] == 1
    skipped = events(run, "skipped")[0]
    assert skipped["url"].endswith("/private/roadmap") and "robots" in skipped["detail"]["error"]
    assert app.parts.index.store.counts()["documents"] == 4
    assert "Secret" not in index_texts(app)


def test_unchanged_pages_cost_a_304_and_no_embeddings(app, site):
    src = add_sitemap_source(app, site)
    sync_now(app.parts, src["id"])
    before = app.embedder.embedded
    run = sync_now(app.parts, src["id"])
    assert run["stats"]["unchanged"] == 4 and run["stats"]["not_modified_304"] == 4
    assert app.embedder.embedded == before and run["stats"]["bytes_downloaded"] == 0


def test_change_reembeds_only_changed_chunks_and_keeps_document_id(app, site):
    src = add_sitemap_source(app, site)
    sync_now(app.parts, src["id"])
    ids = app.parts.index.document_ids()
    pricing_url = site.base + "/docs/pricing"
    pricing_id = ids[pricing_url]
    chunks_before = len(app.parts.index.store.chunks_of(pricing_id))
    assert chunks_before >= 2

    site.put("/docs/pricing", "Pricing - Acme Cloud", PRICING_V2)
    before = app.embedder.embedded
    run = sync_now(app.parts, src["id"])
    upd = events(run, "updated")
    assert len(upd) == 1 and upd[0]["url"] == pricing_url
    d = upd[0]["detail"]
    assert d["embedded"] == 1 and d["reused"] == chunks_before - 1  # only the chunk with the price changed
    assert app.embedder.embedded - before == 1
    assert "- The Pro plan costs $20 per month" in d["diff"]["sample"][0]
    assert "+ The Pro plan costs $25 per month" in d["diff"]["sample"][1]
    assert app.parts.index.document_ids()[pricing_url] == pricing_id   # updated in place
    assert "$25 per month" in index_texts(app) and "$20 per month" not in index_texts(app)

    page = app.parts.state.page(src["id"], pricing_url)
    versions = app.parts.state.versions(page["id"])
    assert len(versions) == 2 and "$25" in versions[0]["text"] and "$20" in versions[1]["text"]


def test_same_text_with_new_etag_is_not_reindexed(app, site):
    src = add_sitemap_source(app, site)
    sync_now(app.parts, src["id"])
    site.put("/docs/pricing", "Pricing - Acme Cloud", PRICING_V1 + "<script>var build = 2;</script>")  # new bytes, same text
    before = app.embedder.embedded
    run = sync_now(app.parts, src["id"])
    reasons = [e["detail"].get("reason") for e in events(run, "unchanged")]
    assert "content hash unchanged" in reasons and app.embedder.embedded == before


def test_sitemap_lastmod_skips_requests(app, site):
    site.lastmod["/docs/limits"] = "2026-11-01"
    src = add_sitemap_source(app, site)
    sync_now(app.parts, src["id"])
    hits = site.hits["/docs/limits"]
    run = sync_now(app.parts, src["id"])
    assert site.hits["/docs/limits"] == hits  # not even a conditional GET
    assert any(e["detail"].get("reason") == "sitemap lastmod unchanged" for e in events(run, "unchanged"))


def test_deleted_and_unlisted_pages_are_removed_from_the_index(app, site):
    src = add_sitemap_source(app, site)
    sync_now(app.parts, src["id"])
    site.unlisted.add("/docs/changelog")   # dropped from the sitemap
    run = sync_now(app.parts, src["id"])
    assert [e["detail"]["reason"] for e in events(run, "removed")] == ["no longer in the sitemap"]
    assert app.parts.index.store.counts()["documents"] == 3

    src2 = app.parts.state.add_source(name="Pinned", kind="urls", urls=[site.base + "/docs/support"], sitemap_url=None,
                                      schedule_minutes=60, sla_hours=2, max_pages=10)
    sync_now(app.parts, src2["id"])
    site.delete("/docs/support")           # 404 now
    run = sync_now(app.parts, src2["id"])
    assert events(run, "removed")[0]["detail"]["reason"] == "HTTP 404"
    assert app.parts.state.page(src2["id"], site.base + "/docs/support")["status"] == "gone"


def test_fetch_errors_keep_serving_the_last_good_version(app, site):
    src = add_sitemap_source(app, site)
    sync_now(app.parts, src["id"])
    site.broken.add("/docs/pricing")
    run = sync_now(app.parts, src["id"])
    assert run["status"] == "completed" and run["stats"]["errors"] == 1
    assert "$20 per month" in index_texts(app)  # still answerable
    page = app.parts.state.page(src["id"], site.base + "/docs/pricing")
    assert page["status"] == "error" and "500" in page["last_error"]


# --------------------------------------------------------------------------- freshness and alerts
def test_staleness_and_alerts_open_and_resolve(app, site):
    src = add_sitemap_source(app, site, sla_hours=2)
    sync_now(app.parts, src["id"])
    m = metrics(app.parts.state, 3)
    assert m["pages"] == 5 and m["stale"] == 1  # the robots-blocked page was never fetched
    app.parts.state.update_source(src["id"], include="/docs/")
    sync_now(app.parts, src["id"])
    assert metrics(app.parts.state, 3)["fresh_pct"] == 100.0

    later = now() + timedelta(hours=3)        # nobody synced for 3h: past the 2h SLA
    m = metrics(app.parts.state, 3, at=later)
    assert m["stale"] == 4 and m["age_histogram"]["1–6h"] == 4
    opened = evaluate_alerts(app.parts.state, 3, at=later)
    assert [a["kind"] for a in opened] == ["stale_pages"] and opened[0]["severity"] == "critical"
    assert evaluate_alerts(app.parts.state, 3, at=later) == []  # deduplicated while open

    evaluate_alerts(app.parts.state, 3)       # back to the present: fresh again -> resolved
    assert app.parts.state.open_alerts() == []


def test_failing_source_alerts_after_repeated_failures(app, site):
    src = add_sitemap_source(app, site)
    site.broken.add("/sitemap.xml")
    for _ in range(3):
        run = sync_now(app.parts, src["id"])
        assert run["status"] == "failed" and "sitemap" in run["error"]
    assert app.parts.state.source(src["id"])["consecutive_failures"] == 3
    kinds = {a["kind"] for a in app.parts.state.open_alerts()}
    assert "source_failing" in kinds
    site.broken.clear()
    assert sync_now(app.parts, src["id"])["status"] == "completed"
    assert "source_failing" not in {a["kind"] for a in app.parts.state.open_alerts()}


def test_ssrf_guard_blocks_private_sitemaps(make_app, site):
    app = make_app(allow_private_urls=False)
    src = add_sitemap_source(app, site)
    run = sync_now(app.parts, src["id"])
    assert run["status"] == "failed" and app.parts.index.store.counts()["documents"] == 0


def test_overlapping_syncs_are_refused(app, site):
    src = add_sitemap_source(app, site)
    assert app.parts.state.claim_source(src["id"])
    assert sync_now(app.parts, src["id"]) is None
    app.parts.state.release_all()
    assert sync_now(app.parts, src["id"])["status"] == "completed"


# --------------------------------------------------------------------------- time-aware answers
def test_answers_use_the_fresh_version_and_show_as_of(app, site):
    src = add_sitemap_source(app, site, include="/docs/")
    sync_now(app.parts, src["id"])
    site.put("/docs/pricing", "Pricing - Acme Cloud", PRICING_V2)
    sync_now(app.parts, src["id"])
    qa = build_qa_graph(app.parts.settings, app.parts.state, app.parts.index, app.embedder, app.answerer)
    result = qa.invoke({"question": "How much does the Pro plan cost?"})["result"]
    assert "$25 per month" in result["markdown"] and "$20" not in result["markdown"]
    src0 = result["sources"][0]
    assert src0["as_of"] and src0["changed_at"] and src0["stale"] is False and result["all_quotes_verified"]
    sent = app.answerer.last_documents
    assert all("as_of_label" in d for d in sent)


def test_stale_sources_are_flagged_in_answers(app, site):
    src = add_sitemap_source(app, site, include="/docs/", sla_hours=1)
    sync_now(app.parts, src["id"])
    page = app.parts.state.page(src["id"], site.base + "/docs/limits")
    app.parts.state.update_page(page["id"], last_checked_at=now() - timedelta(hours=5))
    qa = build_qa_graph(app.parts.settings, app.parts.state, app.parts.index, app.embedder, app.answerer)
    result = qa.invoke({"question": "What is the upload size limit for a single upload?"})["result"]
    assert result["any_stale"] and "out of date" in result["markdown"]


def test_scheduler_syncs_due_sources_with_postgres_checkpointer(settings, site):
    import time

    from src.agent import Scheduler, build_app
    from tests.fakes import HashEmbedder, QuoteAnswerer
    app = build_app(settings, embedder=HashEmbedder(), answerer=QuoteAnswerer())  # real PostgresSaver
    src = add_sitemap_source(app, site, include="/docs/")
    sched = Scheduler(app.parts)
    assert sched.tick() == 1
    for _ in range(100):
        s = app.parts.state.source(src["id"])
        if not s["syncing"] and s["last_success_at"]:
            break
        time.sleep(0.1)
    assert app.parts.state.runs(src["id"])[0]["trigger"] == "schedule"
    assert app.parts.state.runs(src["id"])[0]["stats"]["added"] == 4
    assert sched.tick() == 0  # next run is an hour away
    sched.stop()
