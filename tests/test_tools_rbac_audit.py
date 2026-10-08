"""Tests for tools, RBAC, and audit log."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import RAGAgent, Principal, Role, AuditLog, ToolRegistry
from src.llm import DemoLLM
from src.retrieval import LocalVectorStore


@pytest.fixture
def agent(tmp_path: Path) -> RAGAgent:
    store = LocalVectorStore(persist_dir=tmp_path / "vs", embedding_model_name="all-MiniLM-L6-v2")
    audit = AuditLog(path=tmp_path / "audit.jsonl")
    return RAGAgent(
        vector_store=store,
        llm=DemoLLM(),
        principal=Principal(name="op", role=Role.OPERATOR),
        audit=audit,
    )


def test_calculator_tool():
    reg = ToolRegistry()
    r = reg.run("calculator", "2 + 3 * 4")
    assert r.ok
    assert "14" in r.output


def test_summarizer_tool():
    reg = ToolRegistry()
    r = reg.run("summarizer", "Reactor R-12 max is 450 C. Shutdown in 90 seconds.")
    assert r.ok
    assert "Summary:" in r.output


def test_rbac_viewer_cannot_ingest(tmp_path: Path):
    store = LocalVectorStore(persist_dir=tmp_path / "vs2", embedding_model_name="all-MiniLM-L6-v2")
    agent = RAGAgent(
        vector_store=store,
        llm=DemoLLM(),
        principal=Principal(name="v", role=Role.VIEWER),
        audit=AuditLog(path=tmp_path / "a2.jsonl"),
    )
    doc = tmp_path / "x.txt"
    doc.write_text("hello", encoding="utf-8")
    with pytest.raises(PermissionError):
        agent.ingest_file(doc)


def test_audit_records_query(agent: RAGAgent, tmp_path: Path):
    agent.query("What is 2+2?")
    entries = agent.audit.recent(20)
    assert any(e.get("event") == "prompt" for e in entries)
    assert any(e.get("event") == "answer" for e in entries)


def test_tool_loop_with_forced_tool_in_demo(agent: RAGAgent):
    out = agent.tools.run("calculator", "10 / 2")
    assert out.ok and "5" in out.output
