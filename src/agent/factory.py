"""Wire up the real components: Postgres state, polite fetcher, Document QA index, Voyage, Claude."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models import ClaudeAnswerer, make_embedder
from ..tools.docqa import load_docqa
from ..tools.fetcher import Fetcher
from ..tools.index import IndexWriter
from ..tools.state import State
from ..utils.config import Settings
from .memory import open_checkpointer
from .sync import Components


@dataclass
class App:
    parts: Components
    embedder: Any
    answerer_factory: Any
    _answerer: Any = field(default=None, repr=False)

    @property
    def answerer(self):
        if self._answerer is None:  # Claude is only needed for /ask, so a missing key doesn't stop syncing
            self._answerer = self.answerer_factory()
        return self._answerer


def build_app(settings: Settings, embedder: Any = None, answerer: Any = None, fetcher: Fetcher | None = None,
              checkpointer: Any = "postgres") -> App:
    dq = load_docqa(str(settings.docqa_path))
    state = State(settings.database_url)
    state.init()
    embedder = embedder or make_embedder(settings.voyage_model)
    index = IndexWriter(settings.index_database_url, str(settings.docqa_path), embedder, settings.embedding_dim,
                        settings.chunk_chars, settings.chunk_overlap)
    if checkpointer == "postgres":
        checkpointer = open_checkpointer(settings.database_url)
    parts = Components(settings, state, fetcher or Fetcher(settings, dq.check_url), index, checkpointer)
    app = App(parts, embedder, lambda: ClaudeAnswerer(settings.model, settings.effort))
    app._answerer = answerer
    return app
