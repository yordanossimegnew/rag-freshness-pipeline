"""A tiny, mutable website for tests and the demo: ETag / Last-Modified, 304s, sitemap, robots.txt."""

from __future__ import annotations

import hashlib
import threading
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Site:
    def __init__(self):
        self.pages: dict[str, str] = {}       # path -> html
        self.modified: dict[str, float] = {}  # path -> unix time
        self.lastmod: dict[str, str] = {}     # path -> sitemap <lastmod> (optional)
        self.broken: set[str] = set()         # paths answering 500
        self.unlisted: set[str] = set()       # served but left out of the sitemap
        self.robots = "User-agent: *\nDisallow: /private/\n"
        self.hits: dict[str, int] = {}
        self.not_modified = 0
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None
        self._clock = 1_780_000_000.0

    def put(self, path: str, title: str, body_html: str, lastmod: str | None = None) -> None:
        with self.lock:
            self._clock += 60
            self.pages[path] = f"<html><head><title>{title}</title></head><body><main>{body_html}</main></body></html>"
            self.modified[path] = self._clock
            if lastmod:
                self.lastmod[path] = lastmod

    def delete(self, path: str) -> None:
        with self.lock:
            self.pages.pop(path, None)
            self.lastmod.pop(path, None)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def sitemap(self) -> str:
        items = []
        for path in sorted(self.pages):
            if path in self.unlisted:
                continue
            lm = f"<lastmod>{self.lastmod[path]}</lastmod>" if path in self.lastmod else ""
            items.append(f"<url><loc>{self.base}{path}</loc>{lm}</url>")
        return ('<?xml version="1.0" encoding="UTF-8"?>'
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">' + "".join(items) + "</urlset>")

    def start(self, port: int = 0) -> "Site":
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, code, body=b"", ctype="text/html; charset=utf-8", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                path = self.path.split("?")[0]
                with site.lock:
                    site.hits[path] = site.hits.get(path, 0) + 1
                    if path == "/robots.txt":
                        return self._send(200, site.robots.encode(), "text/plain")
                    if path == "/sitemap.xml":
                        if "/sitemap.xml" in site.broken:
                            return self._send(500, b"boom")
                        return self._send(200, site.sitemap().encode(), "application/xml")
                    if path in site.broken:
                        return self._send(500, b"upstream error")
                    if path not in site.pages:
                        return self._send(404, b"not found")
                    html = site.pages[path].encode()
                    etag = '"' + hashlib.sha1(html).hexdigest()[:16] + '"'
                    lm = formatdate(site.modified[path], usegmt=True)
                    if self.headers.get("If-None-Match") == etag:
                        site.not_modified += 1
                        return self._send(304, headers={"ETag": etag, "Last-Modified": lm})
                    return self._send(200, html, headers={"ETag": etag, "Last-Modified": lm})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()


PRICING_V1 = """<h1>Pricing</h1>
<h2>Starter</h2><p>The Starter plan is free forever and includes one site, 1 GB of storage and community support.</p>
<h2>Pro</h2><p>The Pro plan costs $20 per month and includes 10 sites, 50 GB of storage and email support.</p>
<h2>Business</h2><p>The Business plan costs $99 per month and includes unlimited sites, 1 TB of storage, SSO and a 99.95% uptime SLA.</p>
<h2>Billing</h2><p>All plans are billed monthly. Annual billing saves 15%. Invoices are emailed on the first of each month.</p>"""

PRICING_V2 = PRICING_V1.replace("costs $20 per month and includes 10 sites", "costs $25 per month and includes 15 sites")

LIMITS = """<h1>Limits</h1>
<h2>Requests</h2><p>The API allows 100 requests per second per project. Bursts up to 300 requests are absorbed for five seconds.</p>
<h2>Uploads</h2><p>A single upload can be at most 5 GB. Larger files must use multipart uploads with parts of 100 MB.</p>"""

CHANGELOG = """<h1>Changelog</h1>
<h2>November 2026</h2><p>Edge functions are generally available in all regions.</p>
<h2>October 2026</h2><p>Added IPv6 support for custom domains.</p>"""

SUPPORT = """<h1>Support</h1><p>Email support answers within one business day. Business customers get a 1 hour response time for urgent issues.</p>"""


def acme_site() -> Site:
    site = Site()
    site.put("/docs/pricing", "Pricing - Acme Cloud", PRICING_V1)
    site.put("/docs/limits", "Limits - Acme Cloud", LIMITS)
    site.put("/docs/changelog", "Changelog - Acme Cloud", CHANGELOG)
    site.put("/docs/support", "Support - Acme Cloud", SUPPORT)
    site.put("/private/roadmap", "Roadmap", "<h1>Internal roadmap</h1><p>Secret.</p>")
    return site
