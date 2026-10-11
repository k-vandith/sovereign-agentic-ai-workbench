from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

from src.retrieval import DocumentChunk, HybridVectorStore, LocalVectorStore, calibrate_relevance_threshold


def test_threshold_calibrator_separates_labelled_dev_scores():
    threshold = calibrate_relevance_threshold([0.72, 0.81, 0.91], [0.02, 0.11, 0.22])
    assert 0.22 < threshold < 0.72


def test_fastembed_missing_uses_cpu_hashed_bm25_fallback(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_DISABLE", "1")
    store = HybridVectorStore(persist_dir=tmp_path / "vectors")
    store.add_documents([
        DocumentChunk(
            id="v417",
            text="Valve V-417 maximum working pressure is 38.5 bar.",
            source="valves.txt",
        )
    ])
    assert "fallback" in store.embedding_status().lower()
    matches = store.search("maximum working pressure for valve V-417", top_k=1)
    assert matches and matches[0][0].source == "valves.txt"


def test_unknown_equipment_identifier_is_not_retrieved(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_DISABLE", "1")
    store = HybridVectorStore(persist_dir=tmp_path / "vectors")
    store.add_documents([
        DocumentChunk(
            id="v417",
            text="Valve V-417 maximum working pressure is 38.5 bar.",
            source="valves.txt",
        ),
        DocumentChunk(
            id="p204",
            text="Pump P-204 flow rate is 120 cubic metres per hour.",
            source="pumps.txt",
        ),
    ])
    assert store.search("maximum working pressure of valve V-999", top_k=5) == []
    assert store.search("flow rate of pump P-999", top_k=5) == []


def test_extra_capitalised_words_do_not_wipe_indexed_id_hits(tmp_path: Path, monkeypatch):
    """'Kilnshire' / 'lot 42' / 'page 4' must not discard a real equipment-ID hit (WB-03)."""
    monkeypatch.setenv("FASTEMBED_DISABLE", "1")
    store = HybridVectorStore(persist_dir=tmp_path / "vectors")
    store.add_documents([
        DocumentChunk(
            id="vx",
            text="Valve VX-8841 maximum working pressure is 27.3 bar.",
            source="manuals/valves.txt",
        )
    ])
    for question in (
        "What is valve VX-8841 maximum pressure for lot 42?",
        "According to the Kilnshire dossier, what is valve VX-8841 maximum pressure?",
        "VX-8841 maximum pressure see page 4",
    ):
        hits = store.search(question, top_k=5)
        assert hits, f"expected a hit for: {question}"
        assert hits[0][0].source == "manuals/valves.txt"
        assert "27.3" in hits[0][0].text


def test_equipment_id_restricts_results_to_matching_chunks(tmp_path: Path, monkeypatch):
    """When the question names an indexed ID, only passages containing it rank (WB-02)."""
    monkeypatch.setenv("FASTEMBED_DISABLE", "1")
    store = HybridVectorStore(persist_dir=tmp_path / "vectors")
    store.add_documents([
        DocumentChunk(id="a", text="Valve VZ-100 maximum pressure is 80.0 bar.", source="conflict/a.txt"),
        DocumentChunk(id="b", text="Valve VQ-303 maximum pressure is 19.8 bar.", source="i18n/jp.txt"),
    ])
    hits = store.search("maximum pressure of valve VQ-303?", top_k=5)
    assert hits
    assert all("VQ-303" in h[0].text or "VQ303" in h[0].text.upper().replace("-", "") for h in hits)
    assert hits[0][0].source == "i18n/jp.txt"
    assert "19.8" in hits[0][0].text


def test_default_calibrated_threshold_is_cached_and_bounded(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FASTEMBED_DISABLE", "1")
    store = HybridVectorStore(persist_dir=tmp_path / "vectors")
    first = store.calibrate_threshold()
    second = store.calibrate_threshold()
    assert 0.0 <= first <= 1.0
    assert first == second


def test_existing_hashed_index_is_reembedded_when_fastembed_becomes_available(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("FASTEMBED_DISABLE", "1")
    path = tmp_path / "persisted"
    initial = HybridVectorStore(persist_dir=path)
    text = "Valve V-417 maximum working pressure is 38.5 bar."
    initial.add_documents([DocumentChunk(id="v417", text=text, source="valves.txt")])
    assert initial._embedding_mode == "hashed"

    calls = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str, providers: list[str]):
            assert model_name == "BAAI/bge-small-en-v1.5"
            assert providers == ["CPUExecutionProvider"]

        def embed(self, passages, batch_size: int = 64):
            calls.extend(passages)
            return iter(LocalVectorStore._hashed_text_embeddings(list(passages)))

        def query_embed(self, queries):
            return iter(LocalVectorStore._hashed_text_embeddings(list(queries)))

    monkeypatch.delenv("FASTEMBED_DISABLE", raising=False)
    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    migrated = HybridVectorStore(persist_dir=path)
    migrated.calibrate_threshold()

    assert migrated._embedding_mode == "semantic"
    assert any(text in passage for passage in calls)
    assert migrated.count() == 1
    matches = migrated.search("maximum working pressure for valve V-417", top_k=1)
    assert matches and matches[0][0].source == "valves.txt"
