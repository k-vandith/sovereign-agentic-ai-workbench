from pathlib import Path

from src.retrieval import DocumentChunk, LocalVectorStore


def test_source_summaries_are_grouped_by_document(tmp_path: Path):
    store = LocalVectorStore(persist_dir=tmp_path / "vectorstore")
    store._chunks = [
        DocumentChunk(id="1", text="The machine must be locked out before repair.", source="manual.txt"),
        DocumentChunk(id="2", text="Maintenance is due after 2000 operating hours.", source="manual.txt"),
        DocumentChunk(id="3", text="Wear eye protection.", source="safety.md"),
    ]
    summaries = store.list_sources()
    assert [item["source"] for item in summaries] == ["manual.txt", "safety.md"]
    assert summaries[0]["chunks"] == 2
    assert "locked out" in summaries[0]["preview"]


def test_hashed_cpu_retrieval_finds_matching_local_passage(tmp_path: Path):
    store = LocalVectorStore(persist_dir=tmp_path / "vectorstore")
    store.add_documents([
        DocumentChunk(id="safety", text="Maximum allowable temperature for Reactor R-12 is 450 degrees Celsius.", source="safety.txt"),
        DocumentChunk(id="pump", text="Pump P-204 operating pressure is 8 to 12 bar.", source="pump.txt"),
    ])
    hits = store.search("maximum temperature Reactor R-12", top_k=1)
    assert hits
    assert hits[0][0].source == "safety.txt"
    assert "Hashed-text retrieval" in store.embedding_status()


def test_embedding_mode_is_persisted_and_vectors_reload(tmp_path: Path):
    path = tmp_path / "persisted"
    store = LocalVectorStore(persist_dir=path)
    store.add_documents([
        DocumentChunk(id="1", text="Keep the reactor temperature under the documented limit.", source="safety.txt"),
    ])
    assert (path / "embedding_mode.txt").exists()
    loaded = LocalVectorStore(persist_dir=path)
    assert loaded.count() == 1
    assert loaded.search("reactor temperature limit", top_k=1)[0][0].source == "safety.txt"
