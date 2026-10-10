import json
from pathlib import Path

import numpy as np
import pytest

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


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_search_rejects_invalid_top_k(tmp_path: Path, top_k):
    import pytest

    store = LocalVectorStore(persist_dir=tmp_path / "invalid-k")
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search("question", top_k=top_k)


def test_load_rebuilds_missing_or_corrupt_embeddings_from_saved_chunks(tmp_path: Path):
    for suffix in ("missing", "corrupt"):
        path = tmp_path / suffix
        store = LocalVectorStore(persist_dir=path)
        store.add_documents([
            DocumentChunk(id="safety", text="The reactor temperature limit is 450 C.", source="safety.txt"),
        ])
        embedding_path = path / "embeddings.npy"
        if suffix == "missing":
            embedding_path.unlink()
        else:
            embedding_path.write_bytes(b"not a valid numpy file")

        reloaded = LocalVectorStore(persist_dir=path)
        assert reloaded.count() == 1
        hits = reloaded.search("reactor temperature limit", top_k=1)
        assert hits and hits[0][0].source == "safety.txt"


def test_clear_removes_persisted_vectors(tmp_path: Path):
    path = tmp_path / "clear-store"
    store = LocalVectorStore(persist_dir=path)
    store.add_documents([
        DocumentChunk(id="safety", text="The reactor temperature limit is 450 C.", source="safety.txt"),
    ])
    assert (path / "embeddings.npy").exists()

    store.clear()

    assert store.count() == 0
    assert not (path / "embeddings.npy").exists()

def test_manifest_mismatch_rebuilds_vectors_from_current_chunk_text(tmp_path: Path):
    path = tmp_path / "manifest-recovery"
    store = LocalVectorStore(persist_dir=path)
    store._model = False
    store._embedding_mode = "hashed"
    store.add_documents([
        DocumentChunk(id="manual", text="Old maintenance procedure.", source="manual.txt"),
    ])
    manifest_path = path / "index_manifest.json"
    assert manifest_path.exists()

    metadata_path = path / "chunks.json"
    chunks = json.loads(metadata_path.read_text(encoding="utf-8"))
    chunks[0]["text"] = "Replacement procedure with a new phrase."
    metadata_path.write_text(json.dumps(chunks), encoding="utf-8")

    reloaded = LocalVectorStore(persist_dir=path)
    expected = LocalVectorStore._hashed_text_embeddings(["Replacement procedure with a new phrase."])

    assert reloaded._chunks[0].text == "Replacement procedure with a new phrase."
    assert np.allclose(reloaded._embeddings, expected)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["chunks_sha256"]


def test_store_instance_refreshes_after_another_instance_clears(tmp_path: Path):
    path = tmp_path / "shared-store"
    first = LocalVectorStore(persist_dir=path)
    first.add_documents([
        DocumentChunk(id="old", text="Old indexed document.", source="old.txt"),
    ])
    second = LocalVectorStore(persist_dir=path)
    second.clear()

    first.add_documents([
        DocumentChunk(id="new", text="New indexed document.", source="new.txt"),
    ])

    assert first.count() == 1
    assert first._embeddings is not None
    assert first._embeddings.shape[0] == first.count()
    assert first.search("New indexed document", top_k=1)[0][0].source == "new.txt"
    reloaded = LocalVectorStore(persist_dir=path)
    assert reloaded.count() == 1
    assert reloaded._chunks[0].source == "new.txt"


def test_concurrent_store_instances_do_not_overwrite_each_other(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "concurrent-store"
    stores = [LocalVectorStore(persist_dir=path) for _ in range(4)]

    def add(index: int):
        stores[index].add_documents([
            DocumentChunk(id=f"doc-{index}", text=f"Concurrent procedure {index}.", source=f"doc-{index}.txt"),
        ])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(add, range(4)))

    reloaded = LocalVectorStore(persist_dir=path)
    assert reloaded.count() == 4
    assert {chunk.source for chunk in reloaded._chunks} == {
        "doc-0.txt", "doc-1.txt", "doc-2.txt", "doc-3.txt"
    }
    assert reloaded._embeddings is not None
    assert reloaded._embeddings.shape[0] == reloaded.count()
