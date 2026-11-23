"""Model clients (LangChain): Voyage embeddings and the Claude answerer."""

from __future__ import annotations

from typing import Any, Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..prompts import ANSWER_PROMPT

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Answerer(Protocol):
    name: str
    def answer(self, question: str, documents: list[dict]) -> AIMessage: ...


def make_embedder(model: str) -> Embedder:
    from langchain_voyageai import VoyageAIEmbeddings
    return VoyageAIEmbeddings(model=model, batch_size=64)


def document_block(doc: dict) -> dict:
    """A retrieved chunk as a citable Claude document, with its freshness in the title/context."""
    stale = " · STALE" if doc["stale"] else ""
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": doc["text"]},
        "title": f"{doc['label']} (as of {doc['as_of_label']}{stale})",
        "context": f"From {doc['source']}. Content last changed {doc['changed_label']}.",
        "citations": {"enabled": True},
    }


class ClaudeAnswerer:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", max_tokens: int = 4000):
        self.name = model
        self.chat = ChatAnthropic(model=model, max_tokens=max_tokens, output_config={"effort": effort},
                                  betas=[FALLBACK_BETA], model_kwargs={"fallbacks": "default"}, max_retries=3)

    def answer(self, question: str, documents: list[dict]) -> AIMessage:
        content: list[Any] = [document_block(d) for d in documents]
        content.append({"type": "text", "text": question})
        return self.chat.invoke([SystemMessage(ANSWER_PROMPT), HumanMessage(content=content)])
