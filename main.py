"""RAG Freshness Pipeline: FastAPI app (`uvicorn main:app`) and CLI.

    python main.py serve                          # API + background scheduler
    python main.py add-source "Docs" --sitemap https://example.com/sitemap.xml --every 60
    python main.py sync --all                     # sync every source once, now
    python main.py status                         # freshness report; exit 1 if anything is stale
    python main.py ask "What does the Pro plan cost?"
"""

from __future__ import annotations

import argparse
import sys

from src.utils.config import load_settings


def _app():
    from src.api import create_app
    return create_app()


def __getattr__(name):  # `uvicorn main:app` builds the app lazily (the CLI doesn't need it)
    if name == "app":
        return _app()
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="rag-freshness")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the API and the scheduler")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    add = sub.add_parser("add-source", help="register a sitemap or a list of URLs")
    add.add_argument("name")
    add.add_argument("--sitemap")
    add.add_argument("--url", action="append", default=[])
    add.add_argument("--every", type=int, default=60, help="minutes between syncs")
    add.add_argument("--sla", type=int, help="hours before a page counts as stale")
    add.add_argument("--include")
    add.add_argument("--exclude")
    add.add_argument("--max-pages", type=int, default=200)
    sync = sub.add_parser("sync", help="sync sources now")
    sync.add_argument("source", nargs="?", help="source id or name")
    sync.add_argument("--all", action="store_true")
    sub.add_parser("status", help="freshness report (exit 1 if stale pages or failing sources)")
    ask = sub.add_parser("ask", help="ask a question over the fresh index")
    ask.add_argument("question")
    args = parser.parse_args(argv)
    settings = load_settings()

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("main:app", host=args.host, port=args.port)
        return 0

    from src.agent import build_app, build_qa_graph, evaluate_alerts, metrics, sync_now
    from src.agent.freshness import default_sla
    from src.utils import setup_logging
    setup_logging(settings.log_level, settings.log_dir)
    app = build_app(settings)
    st = app.parts.state

    if args.cmd == "add-source":
        if bool(args.sitemap) == bool(args.url):
            parser.error("give either --sitemap or one or more --url")
        for u in [args.sitemap] if args.sitemap else args.url:
            app.parts.fetcher.check_url(u, settings.allow_private_urls)
        src = st.add_source(name=args.name, kind="sitemap" if args.sitemap else "urls", sitemap_url=args.sitemap,
                            urls=args.url, include=args.include, exclude=args.exclude, schedule_minutes=args.every,
                            sla_hours=args.sla or default_sla(args.every), max_pages=args.max_pages)
        print(f"added {src['name']} ({src['id']}): every {src['schedule_minutes']} min, SLA {src['sla_hours']}h")
        return 0

    if args.cmd == "sync":
        sources = st.list_sources()
        if not args.all:
            sources = [s for s in sources if args.source in (s["id"], s["name"])]
            if not sources:
                print(f"no source {args.source!r}", file=sys.stderr)
                return 2
        failed = False
        for src in sources:
            run = sync_now(app.parts, src["id"])
            if run is None:
                print(f"{src['name']}: already syncing")
                continue
            s = run["stats"]
            failed |= run["status"] == "failed"
            print(f"{src['name']}: {run['status']} · +{s.get('added', 0)} added · ~{s.get('updated', 0)} updated · "
                  f"={s.get('unchanged', 0)} unchanged · -{s.get('removed', 0)} removed · {s.get('errors', 0)} errors · "
                  f"{s.get('chunks_embedded', 0)} chunks embedded, {s.get('chunks_reused', 0)} reused"
                  + (f" · {run['error']}" if run.get("error") else ""))
        return 1 if failed else 0

    if args.cmd == "status":
        evaluate_alerts(st, settings.failure_alert_threshold)
        m = metrics(st, settings.failure_alert_threshold)
        print(f"{m['fresh']}/{m['pages']} pages fresh ({m['fresh_pct']}%)")
        for s in m["sources"]:
            print(f"  {s['name']}: {s['fresh']}/{s['pages']} fresh, SLA {s['sla_hours']}h, "
                  f"{s['consecutive_failures']} failed syncs in a row")
        for a in st.open_alerts():
            print(f"  [{a['severity']}] {a['message']}")
        return 1 if m["stale"] or any(s["failing"] for s in m["sources"]) else 0

    if args.cmd == "ask":
        result = build_qa_graph(settings, st, app.parts.index, app.embedder, app.answerer).invoke(
            {"question": args.question})["result"]
        print(result["markdown"])
        for src in result["sources"]:
            print(f"  [{src['n']}] {src['label']} · as of {src['as_of']}" + (" · STALE" if src["stale"] else ""))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
