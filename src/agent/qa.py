"""Time-aware question answering over the freshness-tracked index (LangGraph).

    START -> retrieve -> annotate -> answer -> END

* retrieve - Document QA's hybrid search (pgvector + full text, rank fusion), over-fetching
* annotate - joins each chunk with its page's freshness (as of, last changed, stale vs SLA)
             and gives recently changed pages a small boost before taking the top k
* answer   - Claude answers with the Citations API; each document says "as of <date>", and the
             prompt prefers the most recently changed source when sources disagree
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ..models import Answerer, Embedder
from ..tools.index import IndexWriter
from ..tools.state import State, now
from ..utils.config import Settings
from .freshness import page_freshness

NOT_FOUND = "I couldn't find anything about that in the indexed pages."


class QAState(TypedDict, total=False):
    question: str
    retrieved: list[dict]
    documents: list[dict]
    result: dict


def _label(dt: str | None) -> str:
    return datetime.fromisoformat(dt).strftime("%Y-%m-%d %H:%M UTC") if dt else "unknown"


def build_qa_graph(settings: Settings, state: State, index: IndexWriter, embedder: Embedder, answerer: Answerer):
    from docqa.agent.agent import page_label
    from docqa.agent.citations import parse_cited_answer, render_markdown  # loaded by IndexWriter

    def retrieve(s: QAState) -> dict:
        q = s["question"]
        hits = index.store.hybrid_search(q, embedder.embed_query(q), top_k=settings.top_k * 2, candidates=40)
        return {"retrieved": hits}

    def annotate(s: QAState) -> dict:
        hits = s.get("retrieved") or []
        pages = state.pages_by_document(list({h["document_id"] for h in hits}))
        at = now()
        docs = []
        for h in hits:
            page = pages.get(h["document_id"])
            fresh = page_freshness(page, page["sla_hours"], at) if page else {"stale": False, "content_age_hours": None}
            age_days = (fresh["content_age_hours"] or 24 * 365) / 24
            boost = 1 + settings.recency_weight * math.exp(-age_days / 30)  # pages changed this month rank a bit higher
            docs.append({**h, "label": page_label(h["title"], h["page_start"], h["page_end"], h["heading"]),
                         "as_of": page and page["last_checked_at"], "changed_at": page and page["last_changed_at"],
                         "as_of_label": _label(page and page["last_checked_at"]),
                         "changed_label": _label(page and page["last_changed_at"]),
                         "stale": fresh["stale"], "tracked": page is not None,
                         "source_name": page and page["source_name"], "rank_score": h["score"] * boost})
        docs.sort(key=lambda d: d["rank_score"], reverse=True)
        return {"documents": docs[: settings.top_k]}

    def answer(s: QAState) -> dict:
        docs = s.get("documents") or []
        if not docs:
            parsed = {"text": NOT_FOUND, "segments": [{"text": NOT_FOUND, "refs": []}], "sources": [],
                      "citations": [], "all_quotes_verified": True}
        else:
            parsed = parse_cited_answer(answerer.answer(s["question"], docs).content, docs)
        by_chunk = {d["id"]: d for d in docs}
        for src in parsed["sources"]:
            d = by_chunk[src["chunk_id"]]
            src.update(as_of=d["as_of"], changed_at=d["changed_at"], stale=d["stale"], source_name=d["source_name"])
        parsed["markdown"] = render_markdown(parsed)
        parsed["any_stale"] = any(src["stale"] for src in parsed["sources"])
        parsed["question"] = s["question"]
        return {"result": parsed}

    g = StateGraph(QAState)
    g.add_node("retrieve", retrieve)
    g.add_node("annotate", annotate)
    g.add_node("answer", answer)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "annotate")
    g.add_edge("annotate", "answer")
    g.add_edge("answer", END)
    return g.compile()
