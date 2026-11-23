"""Incremental updates to Document QA's pgvector index (chunk-level diff).

When a page changes:
  1. it is re-chunked with Document QA's own chunker (identical to its ingestion);
  2. each chunk's embedding text is hashed; chunks whose text already exists in the index
     REUSE their stored vector, and only new text is sent to the embedder;
  3. the document row is updated in place (same id, so links and sessions stay valid) and
     its chunks are replaced in one transaction: readers never see a half-updated page.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, insert, select, update

from .docqa import load_docqa


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


@dataclass
class UpsertResult:
    document_id: str
    created: bool
    chunks: int
    reused: int
    embedded: int
    removed: int
    embed_tokens: int


class IndexWriter:
    def __init__(self, index_database_url: str, docqa_path: str, embedder: Any, dim: int, chunk_chars: int,
                 chunk_overlap: int):
        self.dq = load_docqa(str(docqa_path))
        self.store = self.dq.DocumentStore(index_database_url, dim)
        self.store.init()
        self.docs, self.chunks = self.store.documents, self.store.chunks
        self.embedder, self.chunk_chars, self.chunk_overlap = embedder, chunk_chars, chunk_overlap

    def parse(self, url: str, body: bytes, content_type: str):
        """Turn a fetched page into Document QA's LoadedDocument (HTML or PDF)."""
        if content_type == "application/pdf" or url.lower().endswith(".pdf"):
            doc = self.dq.parse_pdf(url.rsplit("/", 1)[-1].rsplit(".", 1)[0] or url, url, body)
        else:
            doc = self.dq.parse_html(body.decode("utf-8", errors="replace"), url, url, "url", fallback_title=url)
        doc.source = url
        return doc

    @staticmethod
    def embed_text(title: str, chunk) -> str:
        return f"{title}\n{chunk.heading or ''}\n{chunk.text}".strip()  # exactly what Document QA embeds

    def upsert(self, url: str, doc) -> UpsertResult:
        new_chunks = self.dq.chunk_sections(doc.sections, self.chunk_chars, self.chunk_overlap)
        texts = [self.embed_text(doc.title, c) for c in new_chunks]
        hashes = [text_hash(t) for t in texts]

        with self.store.engine.connect() as c:
            row = c.execute(select(self.docs.c.id, self.docs.c.title).where(self.docs.c.source == url)).first()
            existing: dict[str, Any] = {}
            old_count = 0
            if row:
                for ch in c.execute(select(self.chunks.c.text, self.chunks.c.heading, self.chunks.c.embedding)
                                    .where(self.chunks.c.document_id == row.id)):
                    old_count += 1
                    existing[text_hash(f"{row.title}\n{ch.heading or ''}\n{ch.text}".strip())] = ch.embedding

        missing = [i for i, h in enumerate(hashes) if h not in existing]
        fresh = self.embedder.embed_documents([texts[i] for i in missing]) if missing else []
        vectors = [None] * len(texts)
        for i, h in enumerate(hashes):
            if h in existing:
                vectors[i] = list(existing[h])
        for i, v in zip(missing, fresh):
            vectors[i] = v

        doc_id = row.id if row else uuid.uuid4().hex[:12]
        values = dict(title=doc.title[:500], source_type=doc.source_type, source=url, mime=doc.mime, pages=doc.pages,
                      chars=doc.chars, chunks=len(new_chunks), raw=doc.raw if doc.mime == "application/pdf" else None)
        with self.store.engine.begin() as c:  # one transaction: readers see the old or the new page, never half
            if row:
                c.execute(update(self.docs).where(self.docs.c.id == doc_id).values(**values))
                c.execute(delete(self.chunks).where(self.chunks.c.document_id == doc_id))
            else:
                c.execute(insert(self.docs).values(id=doc_id, meta={}, created_at=datetime.now(timezone.utc), **values))
            if new_chunks:
                c.execute(insert(self.chunks), [
                    {"id": f"{doc_id}-{ch.ord}", "document_id": doc_id, "ord": ch.ord, "text": ch.text,
                     "page_start": ch.page_start, "page_end": ch.page_end, "heading": ch.heading, "embedding": v}
                    for ch, v in zip(new_chunks, vectors)])
        reused = len(texts) - len(missing)
        return UpsertResult(doc_id, row is None, len(new_chunks), reused, len(missing),
                            max(0, old_count - reused), sum(len(texts[i]) // 4 for i in missing))

    def remove(self, url: str) -> bool:
        with self.store.engine.begin() as c:
            return bool(c.execute(delete(self.docs).where(self.docs.c.source == url)).rowcount)

    def document_ids(self) -> dict[str, str]:
        with self.store.engine.connect() as c:
            return {r.source: r.id for r in c.execute(select(self.docs.c.id, self.docs.c.source))}
