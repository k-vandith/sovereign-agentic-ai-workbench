"""Print the active backend's threshold learned from independent labelled dev data."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.retrieval import HybridVectorStore


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="workbench-threshold-calibration-") as directory:
        store = HybridVectorStore(
            persist_dir=Path(directory) / "vectors",
            embedding_model_name="BAAI/bge-small-en-v1.5",
        )
        threshold = store.calibrate_threshold()
        print(f"Embedding backend: {store.embedding_status()}")
        print("Calibration dev set: 8 relevant questions + 8 out-of-scope questions")
        print(f"Calibrated relevance threshold: {threshold:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
