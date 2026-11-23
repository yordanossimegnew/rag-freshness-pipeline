"""Polite, conditional HTTP fetching and sitemap parsing.

* Conditional GET: sends If-None-Match / If-Modified-Since, so unchanged pages cost a 304.
* robots.txt is honoured (cached per host); requests to one host are spaced out.
* Redirects are followed by hand and every hop passes Document QA's SSRF check.
"""

from __future__ import annotations

import threading
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx

from ..utils.config import Settings


@dataclass
class FetchResult:
    status: str                 # changed | unchanged | gone | error | blocked
    url: str
    http_status: int | None = None
    body: bytes = b""
    content_type: str = ""
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None


class Fetcher:
    def __init__(self, settings: Settings, check_url, client: httpx.Client | None = None):
        self.s = settings
        self.check_url = check_url
        self.client = client or httpx.Client(timeout=settings.request_timeout,
                                             headers={"User-Agent": settings.user_agent})
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last_hit: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ politeness
    def _wait_turn(self, host: str) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._last_hit.get(host, 0) + self.s.per_host_delay - now
            self._last_hit[host] = max(now, self._last_hit.get(host, 0) + self.s.per_host_delay)
        if wait > 0:
            time.sleep(wait)

    def allowed(self, url: str) -> bool:
        if not self.s.respect_robots:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            parser = urllib.robotparser.RobotFileParser()
            try:
                r = self.client.get(base + "/robots.txt", follow_redirects=True)
                if r.status_code >= 400:
                    parser = None  # no robots.txt: everything allowed
                else:
                    parser.parse(r.text.splitlines())
            except httpx.HTTPError:
                parser = None
            self._robots[base] = parser
        parser = self._robots[base]
        return parser is None or parser.can_fetch(self.s.user_agent, url)

    # ------------------------------------------------------------------ fetching
    def fetch(self, url: str, etag: str | None = None, last_modified: str | None = None) -> FetchResult:
        try:
            self.check_url(url, self.s.allow_private_urls)
        except ValueError as exc:
            return FetchResult("blocked", url, error=str(exc))
        if not self.allowed(url):
            return FetchResult("blocked", url, error="disallowed by robots.txt")
        headers = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        current = url
        try:
            for _ in range(6):
                self._wait_turn(urlparse(current).netloc)
                r = self.client.get(current, headers=headers, follow_redirects=False)
                if r.is_redirect and r.headers.get("location"):
                    current = str(r.url.join(r.headers["location"]))
                    self.check_url(current, self.s.allow_private_urls)  # never redirect into the private network
                    continue
                break
            else:
                return FetchResult("error", url, error="too many redirects")
        except ValueError as exc:
            return FetchResult("blocked", url, error=str(exc))
        except httpx.HTTPError as exc:
            return FetchResult("error", url, error=f"{type(exc).__name__}: {exc}")

        meta = {"etag": r.headers.get("etag"), "last_modified": r.headers.get("last-modified")}
        if r.status_code == 304:
            return FetchResult("unchanged", url, 304, etag=meta["etag"] or etag,
                               last_modified=meta["last_modified"] or last_modified)
        if r.status_code in (404, 410):
            return FetchResult("gone", url, r.status_code)
        if r.status_code >= 400:
            return FetchResult("error", url, r.status_code, error=f"HTTP {r.status_code}")
        if len(r.content) > self.s.max_page_bytes:
            return FetchResult("error", url, r.status_code, error="page too large")
        return FetchResult("changed", url, r.status_code, body=r.content,
                           content_type=r.headers.get("content-type", "").split(";")[0].strip().lower(), **meta)


# ----------------------------------------------------------------------------- sitemaps
_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def parse_sitemap(xml: bytes, base_url: str) -> tuple[list[str], list[dict]]:
    """Returns (child sitemap URLs, page entries [{url, lastmod}])."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"invalid sitemap XML: {exc}") from exc
    tag = root.tag.replace(_NS, "")
    locs = []
    for node in root:
        loc = node.find(f"{_NS}loc")
        if loc is None:
            loc = node.find("loc")
        if loc is None or not (loc.text or "").strip():
            continue
        lastmod = node.find(f"{_NS}lastmod")
        if lastmod is None:
            lastmod = node.find("lastmod")
        locs.append({"url": urljoin(base_url, loc.text.strip()),
                     "lastmod": lastmod.text.strip() if lastmod is not None and lastmod.text else None})
    if tag == "sitemapindex":
        return [e["url"] for e in locs], []
    return [], locs


def discover_sitemap(fetcher: Fetcher, sitemap_url: str, max_pages: int, depth: int = 0) -> list[dict]:
    result = fetcher.fetch(sitemap_url)
    if result.status != "changed":
        raise ValueError(f"could not fetch sitemap {sitemap_url}: {result.error or result.status}")
    children, pages = parse_sitemap(result.body, sitemap_url)
    if depth < 2:
        for child in children:
            if len(pages) >= max_pages:
                break
            pages += discover_sitemap(fetcher, child, max_pages - len(pages), depth + 1)
    seen, unique = set(), []
    for p in pages:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)
    return unique[:max_pages]
