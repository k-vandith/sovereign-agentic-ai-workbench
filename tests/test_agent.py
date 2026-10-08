"""Tests for RAG agent and vector store (demo mode, no external LLM)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent import RAGAgent
from src.llm import DemoLLM
from src.retrieval import LocalVectorStore


@pytest.fixture
def agent(tmp_path: Path) -> RAGAgent:
    store = LocalVectorStore(persist_dir=tmp_path / "vs", embedding_model_name="all-MiniLM-L6-v2")
    return RAGAgent(vector_store=store, llm=DemoLLM())


def test_ingest_and_query(agent: RAGAgent, tmp_path: Path):
    doc = tmp_path / "safety.txt"
    doc.write_text(
        "Maximum allowable temperature for Reactor R-12 is 450 degrees Celsius. "
        "Emergency shutdown requires evacuation within 90 seconds.",
        encoding="utf-8",
    )
    result = agent.ingest_file(doc)
    assert result["chunks_added"] >= 1
    assert agent.store.count() >= 1

    answer = agent.query("What is the max temperature for Reactor R-12?")
    assert "answer" in answer
    assert answer["backend"] == "DemoLLM"
    assert len(answer["sources"]) >= 1
    assert "450" in answer["answer"] or "Reactor" in answer["answer"]


def test_empty_kb_query(agent: RAGAgent):
    result = agent.query("What is the meaning of life?")
    assert "answer" in result
    assert result["conversation_id"]


def test_conversation_tracking(agent: RAGAgent):
    r1 = agent.query("Hello")
    cid = r1["conversation_id"]
    r2 = agent.query("Follow-up", conversation_id=cid)
    assert r2["conversation_id"] == cid
    convs = agent.list_conversations()
    assert any(c["id"] == cid for c in convs)
