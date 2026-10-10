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
    document = tmp_path / "arithmetic.txt"
    document.write_text("The arithmetic example says two plus two equals four.", encoding="utf-8")
    agent.ingest_file(document)
    agent.settings.relevance_threshold = -1.0
    agent.query("What does the arithmetic example say about two plus two?")
    entries = agent.audit.recent(20)
    assert any(e.get("event") == "prompt" for e in entries)
    assert any(e.get("event") == "answer" for e in entries)


def test_tool_loop_with_forced_tool_in_demo(agent: RAGAgent):
    out = agent.tools.run("calculator", "10 / 2")
    assert out.ok and "5" in out.output


def test_audit_recent_zero_limit_returns_no_records(tmp_path: Path):
    audit = AuditLog(path=tmp_path / "audit-limit.jsonl")
    audit.record("one")
    audit.record("two")

    assert audit.recent(limit=0) == []
    assert audit.recent(limit=-1) == []
    assert [entry["event"] for entry in audit.recent(limit=1)] == ["two"]


def test_calculator_rejects_excessive_exponent_expression_and_ast_size():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="Exponent magnitude"):
        registry.calculator("2 ** 101")
    with pytest.raises(ValueError, match="256 characters"):
        registry.calculator("1" * 257)
    with pytest.raises(ValueError, match="64 syntax nodes"):
        registry.calculator(" + ".join(["1"] * 40))


def test_image_analysis_only_reads_images_inside_allowed_roots(tmp_path: Path):
    from PIL import Image

    allowed = tmp_path / "data"
    allowed.mkdir()
    safe_image = allowed / "safe.png"
    Image.new("RGB", (3, 4)).save(safe_image)

    outside_image = tmp_path / "outside.png"
    Image.new("RGB", (10, 10)).save(outside_image)

    registry = ToolRegistry(allowed_image_roots=[allowed])
    safe_result = registry.image_analysis(str(safe_image))
    outside_result = registry.image_analysis(str(outside_image))

    assert "3x4 px" in safe_result
    assert "restricted to the configured local data directories" in outside_result


def test_image_analysis_rejects_symlink_outside_allowed_root(tmp_path: Path):
    from PIL import Image

    allowed = tmp_path / "data"
    allowed.mkdir()
    outside_image = tmp_path / "outside.png"
    Image.new("RGB", (2, 2)).save(outside_image)
    link = allowed / "linked.png"
    try:
        link.symlink_to(outside_image)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not available in this environment.")

    result = ToolRegistry(allowed_image_roots=[allowed]).image_analysis(str(link))

    assert "restricted to the configured local data directories" in result



def test_audit_hash_chain_verifies_and_detects_tampering(tmp_path: Path):
    import json

    audit = AuditLog(path=tmp_path / "chained-audit.jsonl")
    audit.record("first", value="one")
    audit.record("second", value="two")

    verified = audit.verify()
    assert verified["valid"] is True
    assert verified["checked_entries"] == 2
    assert verified["legacy_entries"] == 0

    lines = audit.path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["value"] = "tampered"
    lines[0] = json.dumps(first, ensure_ascii=False)
    audit.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    invalid = audit.verify()
    assert invalid["valid"] is False
    with pytest.raises(RuntimeError, match="integrity check failed"):
        audit.record("third")


def test_audit_chain_anchors_legacy_prefix_without_claiming_old_hashes(tmp_path: Path):
    import json

    path = tmp_path / "legacy-audit.jsonl"
    path.write_text(json.dumps({"event": "legacy", "message": "old entry"}) + "\n", encoding="utf-8")
    audit = AuditLog(path=path)

    before = audit.verify()
    assert before["valid"] is True
    assert before["legacy_entries"] == 1
    assert before["checked_entries"] == 0

    audit.record("upgraded")
    after = audit.verify()
    assert after["valid"] is True
    assert after["legacy_entries"] == 1
    assert after["checked_entries"] == 1


def test_agent_audit_verification_requires_admin_role(tmp_path: Path):
    store = LocalVectorStore(persist_dir=tmp_path / "audit-permission-vs")
    agent = RAGAgent(
        vector_store=store,
        llm=DemoLLM(),
        principal=Principal(name="operator", role=Role.OPERATOR),
        audit=AuditLog(path=tmp_path / "protected-audit.jsonl"),
    )
    with pytest.raises(PermissionError):
        agent.verify_audit()


def test_audit_chain_remains_valid_with_concurrent_instances(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "concurrent-audit.jsonl"
    logs = [AuditLog(path=path) for _ in range(4)]

    def append_events(worker: int):
        for sequence in range(10):
            logs[worker].record("concurrent_event", worker=worker, sequence=sequence)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(append_events, range(4)))

    result = AuditLog(path=path).verify()
    assert result["valid"] is True
    assert result["checked_entries"] == 40
    assert result["legacy_entries"] == 0
