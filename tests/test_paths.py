"""Filename safety and knowledge-base reset."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import RAGAgent, Principal, Role, AuditLog
from src.llm import DemoLLM
from src.paths import safe_filename
from src.retrieval import LocalVectorStore


def test_safe_filename_strips_traversal():
    assert safe_filename("../../notes.txt") == "notes.txt"
    assert safe_filename(r"..\\secret.md") == "secret.md"


def test_safe_filename_rejects_empty():
    with pytest.raises(ValueError):
        safe_filename("..")


def test_clear_knowledge_base(tmp_path: Path):
    store = LocalVectorStore(persist_dir=tmp_path / "vs", embedding_model_name="all-MiniLM-L6-v2")
    agent = RAGAgent(
        vector_store=store,
        llm=DemoLLM(),
        principal=Principal(name="op", role=Role.OPERATOR),
        audit=AuditLog(path=tmp_path / "a.jsonl"),
    )
    doc = tmp_path / "note.txt"
    doc.write_text("Reactor limit is 450 C.", encoding="utf-8")
    agent.ingest_file(doc)
    assert agent.store.count() >= 1
    agent.clear_knowledge_base()
    assert agent.store.count() == 0


def test_ui_module_has_main():
    import src.ui.app as ui

    assert callable(ui.main)
