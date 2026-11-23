from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import create_app
from tests.site import PRICING_V2


def client(settings, app):
    return TestClient(create_app(settings, app_parts=app, run_inline=True, scheduler=False))


def test_api_flow(settings, app, site):
    with client(settings, app) as c:
        assert c.get("/health").json()["answer_model"].startswith("scripted")
        r = c.post("/sources", json={"name": "Acme", "sitemap_url": site.base + "/sitemap.xml", "include": "/docs/",
                                     "schedule_minutes": 30})
        assert r.status_code == 201, r.text
        src = r.json()
        assert src["sla_hours"] == 1  # two missed 30-minute syncs
        run = c.post(f"/sources/{src['id']}/sync").json()
        assert run["stats"]["added"] == 4

        site.put("/docs/pricing", "Pricing - Acme Cloud", PRICING_V2)
        run = c.post(f"/sources/{src['id']}/sync").json()
        assert run["stats"]["updated"] == 1
        assert [r["id"] for r in c.get("/runs", params={"source_id": src["id"]}).json()][0] == run["id"]

        pages = c.get("/pages", params={"source_id": src["id"]}).json()
        pricing = next(p for p in pages if p["url"].endswith("/pricing"))
        assert pricing["stale"] is False
        detail = c.get(f"/pages/{pricing['id']}").json()
        assert len(detail["versions"]) == 2

        m = c.get("/metrics").json()
        assert m["fresh_pct"] == 100.0 and m["sources"][0]["pages"] == 4
        assert c.get("/sources").json()[0]["freshness"]["fresh"] == 4

        ans = c.post("/ask", json={"question": "How much does the Pro plan cost?"}).json()
        assert "$25" in ans["markdown"] and ans["sources"][0]["as_of"]

        c.delete(f"/sources/{src['id']}")
        assert app.parts.index.store.counts()["documents"] == 0


def test_api_validation_and_conflicts(settings, app, site):
    with client(settings, app) as c:
        assert c.post("/sources", json={"name": "x"}).status_code == 422
        assert c.post("/sources", json={"name": "x", "sitemap_url": site.base + "/sitemap.xml",
                                        "urls": [site.base + "/a"]}).status_code == 422
        src = c.post("/sources", json={"name": "x", "urls": [site.base + "/docs/support"]}).json()
        app.parts.state.claim_source(src["id"])
        assert c.post(f"/sources/{src['id']}/sync").status_code == 409
        assert c.get("/sources/nope").status_code == 404


def test_api_rejects_private_urls_by_default(make_app, settings, site):
    from dataclasses import replace
    strict = replace(settings, allow_private_urls=False)
    with client(strict, make_app(allow_private_urls=False)) as c:
        r = c.post("/sources", json={"name": "x", "sitemap_url": site.base + "/sitemap.xml"})
        assert r.status_code == 422


def test_cli_status_exit_code(app, site, monkeypatch, capsys):
    import main
    from src.agent import sync_now
    src = app.parts.state.add_source(name="Acme", kind="sitemap", sitemap_url=site.base + "/sitemap.xml", urls=[],
                                     include="/docs/", schedule_minutes=60, sla_hours=2, max_pages=50)
    sync_now(app.parts, src["id"])
    monkeypatch.setattr("src.agent.build_app", lambda s: app)
    assert main.main(["status"]) == 0
    assert "4/4 pages fresh" in capsys.readouterr().out
    site.broken.add("/sitemap.xml")
    app.parts.state.update_source(src["id"], consecutive_failures=2)
    assert main.main(["sync", "Acme"]) == 1
    assert main.main(["status"]) == 1
