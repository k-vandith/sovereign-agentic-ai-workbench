"""CPU-only local text retrieval with optional transformer embeddings and FAISS."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single indexed text chunk with metadata."""

    id: str
    text: str
    source: str
    page: int | None = None
    metadata: dict[str, Any] | None = None


class LocalVectorStore:
    """Local vector store with a dependency-light hashed-text default.

    If sentence-transformers is installed and its model is available, semantic
    embeddings are used. Otherwise deterministic feature hashing supports lexical
    retrieval without a model download. FAISS is optional; NumPy cosine search
    remains available as a fallback.
    """

    def __init__(
        self,
        persist_dir: Path,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name
        self._model = None
        self._semantic_status = "not checked"
        self._index = None
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None
        self._use_faiss = False
        self._load()

    @staticmethod
    def _hashed_text_embeddings(texts: list[str], dimensions: int = 768) -> np.ndarray:
        """Create deterministic CPU-only token and bigram vectors without model files."""
        matrix = np.zeros((len(texts), dimensions), dtype=np.float32)
        for row_index, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9][a-z0-9._-]*", str(text).lower())
            features = tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "little") % dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row_index, slot] += sign
            norm = float(np.linalg.norm(matrix[row_index]))
            if norm:
                matrix[row_index] /= norm
        return matrix

    def _get_model(self):
        if self._model is False:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self._model = False
                self._semantic_status = "not installed"
                logger.info("sentence-transformers is not installed; using hashed-text retrieval.")
                return None
            try:
                logger.info("Loading embedding model: %s", self.embedding_model_name)
                self._model = SentenceTransformer(self.embedding_model_name, device="cpu")
                self._semantic_status = "active"
            except Exception as exc:
                self._model = False
                self._semantic_status = "unavailable; using hashed-text retrieval"
                logger.warning("Semantic embedding model unavailable; using hashed-text retrieval: %s", exc)
                return None
        return self._model

    def embedding_status(self) -> str:
        """Describe which local embedding mode is currently active."""
        if self._semantic_status == "not checked":
            if self._model is False:
                return "Hashed-text retrieval (CPU)"
            if self._model is not None:
                return "Semantic embeddings (CPU)"
            try:
                import importlib.util
                if importlib.util.find_spec("sentence_transformers") is None:
                    self._semantic_status = "not installed"
            except (ImportError, ValueError):
                self._semantic_status = "not installed"
        if self._semantic_status == "active":
            return "Semantic embeddings (CPU)"
        return "Hashed-text retrieval (CPU)"

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        if model is None:
            return self._hashed_text_embeddings(texts)
        try:
            emb = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, device="cpu")
            return np.asarray(emb, dtype=np.float32)
        except Exception as exc:
            logger.warning("Semantic embedding failed; switching to hashed-text retrieval: %s", exc)
            self._model = False
            self._semantic_status = "unavailable; using hashed-text retrieval"
            return self._hashed_text_embeddings(texts)

    def add_documents(self, chunks: list[DocumentChunk]) -> int:
        """Add chunks to the store and rebuild the index."""
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        new_emb = self._embed(texts)
        start_idx = len(self._chunks)
        for i, c in enumerate(chunks):
            c.id = c.id or f"chunk_{start_idx + i}"
            self._chunks.append(c)
        if self._embeddings is None:
            self._embeddings = new_emb
        else:
            self._embeddings = np.vstack([self._embeddings, new_emb])
        self._rebuild_index()
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Return top-k most similar chunks with cosine similarity scores."""
        if not self._chunks or self._embeddings is None:
            return []
        q_emb = self._embed([query])[0]
        if self._use_faiss and self._index is not None:

            scores, indices = self._index.search(
                q_emb.reshape(1, -1).astype(np.float32), min(top_k, len(self._chunks))
            )
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append((self._chunks[idx], float(score)))
            return results
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(q_emb)
        norms = np.where(norms == 0, 1e-9, norms)
        sims = (self._embeddings @ q_emb) / norms
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self._chunks[i], float(sims[i])) for i in top_idx]

    def clear(self) -> None:
        self._chunks = []
        self._embeddings = None
        self._index = None
        self._save()

    def list_sources(self) -> list[dict[str, Any]]:
        """Return a safe summary of documents represented in the local index."""
        summaries: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            name = Path(chunk.source).name or "Untitled document"
            item = summaries.setdefault(
                name,
                {"source": name, "chunks": 0, "preview": "", "pages": set()},
            )
            item["chunks"] += 1
            if not item["preview"] and chunk.text.strip():
                item["preview"] = " ".join(chunk.text.split())[:220]
            if chunk.page is not None:
                item["pages"].add(chunk.page)
        result = []
        for item in summaries.values():
            item["pages"] = len(item["pages"])
            result.append(item)
        return sorted(result, key=lambda item: item["source"].casefold())

    def count(self) -> int:
        return len(self._chunks)

    def _rebuild_index(self) -> None:
        if self._embeddings is None or len(self._embeddings) == 0:
            return
        try:
            import faiss

            dim = self._embeddings.shape[1]
            faiss.normalize_L2(self._embeddings)
            index = faiss.IndexFlatIP(dim)
            index.add(self._embeddings)
            self._index = index
            self._use_faiss = True
        except Exception as exc:
            logger.warning("FAISS unavailable, using numpy fallback: %s", exc)
            self._use_faiss = False
            self._index = None

    def _save(self) -> None:
        meta_path = self.persist_dir / "chunks.json"
        emb_path = self.persist_dir / "embeddings.npy"
        data = [asdict(c) for c in self._chunks]
        meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        if self._embeddings is not None:
            np.save(str(emb_path), self._embeddings)

    def _load(self) -> None:
        meta_path = self.persist_dir / "chunks.json"
        emb_path = self.persist_dir / "embeddings.npy"
        if meta_path.exists():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            self._chunks = [DocumentChunk(**item) for item in raw]
        if emb_path.exists() and self._chunks:
            self._embeddings = np.load(str(emb_path))
            self._rebuild_index()
        logger.info("Loaded vector store with %d chunks", len(self._chunks))
