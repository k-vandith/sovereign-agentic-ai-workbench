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
