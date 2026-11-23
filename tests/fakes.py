"""Deterministic stand-ins for Voyage and Claude."""

from __future__ import annotations

import hashlib
import math
import re

from langchain_core.messages import AIMessage

STOP = {"what", "does", "the", "how", "much", "cost", "is", "are"}


class HashEmbedder:
    """Bag-of-words hashed into 1024 dims; counts how many texts were embedded."""

    def __init__(self, dim: int = 1024):
        self.dim, self.embedded = dim, 0

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9$]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class QuoteAnswerer:
    """Answers by quoting the sentence of the best document that mentions the question's words."""

    name = "scripted answerer (stand-in for Claude)"

    def __init__(self):
        self.last_documents: list[dict] = []

    def answer(self, question: str, documents: list[dict]) -> AIMessage:
        self.last_documents = documents
        words = [w for w in re.findall(r"[a-z]{3,}", question.lower()) if w not in STOP]

        def hits(s: str) -> int:
            return sum(w in s.lower() for w in words)

        best = max(range(len(documents)), key=lambda i: hits(documents[i]["text"]))
        text = documents[best]["text"]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        quote = max(sentences, key=hits)
        lead = ("This may be out of date (the page is past its freshness SLA), but "
                if documents[best]["stale"] else "According to the current version of the page, ")
        start = text.find(quote)
        return AIMessage(content=[
            {"type": "text", "text": lead},
            {"type": "text", "text": quote[0].lower() + quote[1:],
             "citations": [{"type": "char_location", "document_index": best, "cited_text": quote,
                            "start_char_index": start, "end_char_index": start + len(quote)}]},
        ])
