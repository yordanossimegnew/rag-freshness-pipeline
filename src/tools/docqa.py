"""Import Document QA's real code (parsers, chunker, pgvector store, QA graph) as `docqa`.

The pipeline keeps *that project's* index fresh, so it uses its exact parsing and chunking:
an updated page ends up chunked exactly as if it had been ingested through Document QA.
Keep the repositories side by side, or set DOCQA_PATH.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


class DocqaNotFound(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_docqa(path: str) -> SimpleNamespace:
    root = Path(path).resolve() / "src"
    if not (root / "__init__.py").exists():
        raise DocqaNotFound(f"Document QA project not found at {Path(path).resolve()}. "
                            "Clone document-qa-citations next to this repository or set DOCQA_PATH.")
    if "docqa" not in sys.modules:
        spec = importlib.util.spec_from_file_location("docqa", root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["docqa"] = module
        spec.loader.exec_module(module)

    from docqa.agent import DocumentQA
    from docqa.agent import open_checkpointer as qa_checkpointer
    from docqa.ingest import chunk_sections, loaders
    from docqa.tools.store import DocumentStore
    from docqa.utils.config import Settings as DocqaSettings

    return SimpleNamespace(DocumentStore=DocumentStore, chunk_sections=chunk_sections, parse_html=loaders._html,
                           parse_pdf=loaders._pdf, check_url=loaders.check_url, LoaderError=loaders.LoaderError,
                           DocumentQA=DocumentQA, qa_checkpointer=qa_checkpointer, DocqaSettings=DocqaSettings)
