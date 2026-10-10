"""CPU-only local text retrieval with optional transformer embeddings and FAISS."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, Any] = {}


def _shared_thread_lock(path_key: str):
    with _STORE_LOCKS_GUARD:
        return _STORE_LOCKS.setdefault(path_key, threading.RLock())


@contextmanager
def _exclusive_process_lock(path: Path):
    """Acquire an OS file lock so separate UI/API processes cannot interleave writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
        self._lock = _shared_thread_lock(str(self.persist_dir.resolve()))
        self._loaded_manifest_sha: str | None = None
        self._model = None
        self._semantic_status = "not checked"
        self._embedding_mode: str | None = None
        self._index = None
        self._chunks: list[DocumentChunk] = []
        self._embeddings: np.ndarray | None = None
        self._use_faiss = False
        with self._lock, _exclusive_process_lock(self.persist_dir / ".index.lock"):
            self._load()

    @staticmethod
    def _hashed_text_embeddings(texts: list[str], dimensions: int = 384) -> np.ndarray:
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

    def _rehash_existing_chunks(self) -> None:
        """Rebuild an existing store if its previous semantic backend is no longer available."""
        if not self._chunks:
            self._embedding_mode = "hashed"
            return
        self._embeddings = self._hashed_text_embeddings([chunk.text for chunk in self._chunks])
        self._embedding_mode = "hashed"
        self._semantic_status = "not installed; using hashed-text retrieval"
        self._rebuild_index()
        self._save()

    def _get_model(self):
        if self._embedding_mode == "hashed" or self._model is False:
            return None
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self._model = False
                if self._embedding_mode == "semantic":
                    self._rehash_existing_chunks()
                else:
                    self._embedding_mode = "hashed"
                self._semantic_status = "not installed; using hashed-text retrieval"
                logger.info("sentence-transformers is not installed; using hashed-text retrieval.")
                return None
            try:
                logger.info("Loading embedding model: %s", self.embedding_model_name)
                self._model = SentenceTransformer(self.embedding_model_name, device="cpu")
                self._embedding_mode = "semantic"
                self._semantic_status = "active"
            except Exception as exc:
                self._model = False
                if self._embedding_mode == "semantic":
                    self._rehash_existing_chunks()
                else:
                    self._embedding_mode = "hashed"
                self._semantic_status = "unavailable; using hashed-text retrieval"
                logger.warning("Semantic embedding model unavailable; using hashed-text retrieval: %s", exc)
                return None
        return self._model

    def embedding_status(self) -> str:
        """Describe the embedding mode used by the current local index."""
        if self._embedding_mode == "hashed":
            return "Hashed-text retrieval (CPU)"
        if self._embedding_mode == "semantic":
            if self._semantic_status == "active":
                return "Semantic embeddings (CPU)"
            return "Semantic embeddings (CPU; model loads on first use)"
        if self._model is not None and self._model is not False:
            return "Semantic embeddings (CPU)"
        try:
            import importlib.util
            if importlib.util.find_spec("sentence_transformers") is not None:
                return "Semantic embeddings available (CPU; not loaded)"
        except (ImportError, ValueError):
            pass
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
            if self._embedding_mode == "semantic":
                self._rehash_existing_chunks()
            else:
                self._embedding_mode = "hashed"
            self._semantic_status = "unavailable; using hashed-text retrieval"
            return self._hashed_text_embeddings(texts)

    def add_documents(
        self,
        chunks: list[DocumentChunk],
        *,
        replace_source: str | None = None,
    ) -> int:
        """Serialize updates and refresh stale state before applying new chunks."""
        with self._lock, _exclusive_process_lock(self.persist_dir / ".index.lock"):
            self._refresh_if_changed()
            return self._add_documents_unlocked(chunks, replace_source=replace_source)

    def _add_documents_unlocked(
        self,
        chunks: list[DocumentChunk],
        *,
        replace_source: str | None = None,
    ) -> int:
        """Add chunks, optionally replacing a source without retaining stale passages."""
        if not chunks:
            if replace_source is None:
                return 0
            keep_indices = [
                index for index, chunk in enumerate(self._chunks)
                if chunk.source != replace_source
            ]
            if len(keep_indices) == len(self._chunks):
                return 0
            self._chunks = [self._chunks[index] for index in keep_indices]
            if not self._chunks:
                self._embeddings = None
                self._index = None
                self._use_faiss = False
            elif self._embeddings is not None:
                self._embeddings = self._embeddings[keep_indices]
            else:
                self._embedding_mode = "hashed"
                self._model = False
                self._embeddings = self._hashed_text_embeddings([chunk.text for chunk in self._chunks])
            self._rebuild_index()
            self._save()
            return 0

        new_emb = self._embed([chunk.text for chunk in chunks])
        if replace_source is not None:
            keep_indices = [
                index for index, chunk in enumerate(self._chunks)
                if chunk.source != replace_source
            ]
            retained_chunks = [self._chunks[index] for index in keep_indices]
            retained_embeddings = (
                self._embeddings[keep_indices] if self._embeddings is not None else None
            )
            start_idx = len(retained_chunks)
            for index, chunk in enumerate(chunks):
                chunk.id = chunk.id or f"chunk_{start_idx + index}"
            combined_chunks = retained_chunks + chunks

            if retained_chunks and (
                retained_embeddings is None
                or retained_embeddings.ndim != 2
                or retained_embeddings.shape[1] != new_emb.shape[1]
            ):
                # The configured embedding model may have changed since retained vectors
                # were created. Rebuild all vectors together to preserve dimensionality.
                combined_embeddings = self._embed([chunk.text for chunk in combined_chunks])
            elif retained_chunks and retained_embeddings is not None:
                combined_embeddings = np.vstack([retained_embeddings, new_emb])
            else:
                combined_embeddings = new_emb

            self._chunks = combined_chunks
            self._embeddings = combined_embeddings
        else:
            start_idx = len(self._chunks)
            for index, chunk in enumerate(chunks):
                chunk.id = chunk.id or f"chunk_{start_idx + index}"
            if (
                self._embeddings is not None
                and self._embeddings.ndim == 2
                and self._embeddings.shape[1] != new_emb.shape[1]
            ):
                # Avoid a shape-mismatch crash if the available embedding backend changed.
                combined_chunks = self._chunks + chunks
                self._chunks = combined_chunks
                self._embeddings = self._embed([chunk.text for chunk in combined_chunks])
            else:
                self._chunks.extend(chunks)
                if self._embeddings is None or len(self._embeddings) == 0:
                    self._embeddings = new_emb
                else:
                    self._embeddings = np.vstack([self._embeddings, new_emb])

        self._rebuild_index()
        self._save()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Refresh persistent changes and run retrieval under a shared store lock."""
        with self._lock, _exclusive_process_lock(self.persist_dir / ".index.lock"):
            self._refresh_if_changed()
            return self._search_unlocked(query, top_k=top_k)

    def _search_unlocked(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Return top-k most similar chunks with cosine similarity scores."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
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
        """Clear the shared index as one serialized persistent operation."""
        with self._lock, _exclusive_process_lock(self.persist_dir / ".index.lock"):
            self._chunks = []
            self._embeddings = None
            self._index = None
            self._use_faiss = False
            self._save()

    def list_sources(self) -> list[dict[str, Any]]:
        """Return a safe summary of documents represented in the local index.
        
        Persistent updates from other app processes are picked up before the summary.
        """
        with self._lock, _exclusive_process_lock(self.persist_dir / ".index.lock"):
            self._refresh_if_changed()
            return self._list_sources_unlocked()

    def _list_sources_unlocked(self) -> list[dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for chunk in self._chunks:
            name = Path(chunk.source).as_posix() or "Untitled document"
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
        with self._lock, _exclusive_process_lock(self.persist_dir / ".index.lock"):
            self._refresh_if_changed()
            return len(self._chunks)

    def _rebuild_index(self) -> None:
        if self._embeddings is None or len(self._embeddings) == 0:
            self._index = None
            self._use_faiss = False
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
        """Persist the index atomically and publish a manifest only after all files are ready."""
        meta_path = self.persist_dir / "chunks.json"
        emb_path = self.persist_dir / "embeddings.npy"
        mode_path = self.persist_dir / "embedding_mode.txt"
        model_path = self.persist_dir / "embedding_model_name.txt"
        manifest_path = self.persist_dir / "index_manifest.json"

        _atomic_write_bytes(
            meta_path,
            json.dumps([asdict(chunk) for chunk in self._chunks], indent=2).encode("utf-8"),
        )
        if self._embeddings is not None:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "wb", dir=self.persist_dir, prefix=f".{emb_path.name}.", delete=False
                ) as stream:
                    temporary_path = Path(stream.name)
                    np.save(stream, self._embeddings)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, emb_path)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        else:
            emb_path.unlink(missing_ok=True)

        _atomic_write_bytes(mode_path, (self._embedding_mode or "unknown").encode("utf-8"))
        _atomic_write_bytes(model_path, self.embedding_model_name.encode("utf-8"))
        manifest = {
            "chunks_sha256": _sha256_file(meta_path),
            "embeddings_sha256": _sha256_file(emb_path) if emb_path.exists() else None,
            "embedding_mode_sha256": _sha256_file(mode_path),
            "embedding_model_name_sha256": _sha256_file(model_path),
        }
        _atomic_write_bytes(
            manifest_path,
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
        self._loaded_manifest_sha = _sha256_file(manifest_path)

    def _refresh_if_changed(self) -> None:
        """Reload only when the manifest changed, avoiding stale concurrent writers."""
        manifest_path = self.persist_dir / "index_manifest.json"
        if not manifest_path.exists():
            if self._loaded_manifest_sha is not None:
                self._load()
            return
        current_sha = _sha256_file(manifest_path)
        if current_sha != self._loaded_manifest_sha:
            self._load()

    def _load(self) -> None:
        # Refreshes may follow a clear or replacement from another process.
        self._chunks = []
        self._embeddings = None
        self._index = None
        self._use_faiss = False
        self._embedding_mode = None
        meta_path = self.persist_dir / "chunks.json"
        emb_path = self.persist_dir / "embeddings.npy"
        mode_path = self.persist_dir / "embedding_mode.txt"
        model_path = self.persist_dir / "embedding_model_name.txt"
        manifest_path = self.persist_dir / "index_manifest.json"
        if meta_path.exists():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            self._chunks = [DocumentChunk(**item) for item in raw]
        if mode_path.exists():
            saved_mode = mode_path.read_text(encoding="utf-8").strip()
            self._embedding_mode = saved_mode if saved_mode in {"semantic", "hashed"} else None

        saved_model_name = model_path.read_text(encoding="utf-8").strip() if model_path.exists() else None
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_valid = (
                    manifest.get("chunks_sha256") == _sha256_file(meta_path)
                    and manifest.get("embeddings_sha256") == (
                        _sha256_file(emb_path) if emb_path.exists() else None
                    )
                    and manifest.get("embedding_mode_sha256") == _sha256_file(mode_path)
                    and manifest.get("embedding_model_name_sha256") == _sha256_file(model_path)
                )
            except (OSError, ValueError, TypeError, AttributeError):
                manifest_valid = False
            if not manifest_valid:
                logger.warning("Persisted index files do not match their manifest; rebuilding vectors from text.")
                if self._chunks:
                    self._embedding_mode = "hashed"
                    self._model = False
                    self._rehash_existing_chunks()
                else:
                    self._embeddings = None
                    self._index = None
                    self._use_faiss = False
                    self._save()
                return

        if self._chunks:
            model_changed = (
                self._embedding_mode == "semantic"
                and (
                    saved_model_name is None
                    or saved_model_name != self.embedding_model_name
                )
            )
            if model_changed:
                logger.info("Embedding model changed; rebuilding vectors from stored document text.")
                self._embedding_mode = None
                self._embeddings = self._embed([chunk.text for chunk in self._chunks])
                self._rebuild_index()
                self._save()
            else:
                loaded_embeddings = None
                if emb_path.exists():
                    try:
                        candidate = np.load(str(emb_path), allow_pickle=False)
                        valid = (
                            candidate.ndim == 2
                            and candidate.shape[0] == len(self._chunks)
                            and candidate.shape[1] > 0
                            and np.issubdtype(candidate.dtype, np.number)
                            and np.isfinite(candidate).all()
                        )
                        if valid:
                            loaded_embeddings = candidate.astype(np.float32, copy=False)
                    except (OSError, ValueError, EOFError) as exc:
                        logger.warning(
                            "Persisted embeddings could not be loaded; rebuilding from text (%s).",
                            type(exc).__name__,
                        )
                if loaded_embeddings is None:
                    logger.warning("Persisted embeddings are missing or incompatible; rebuilding from text.")
                    self._embedding_mode = "hashed"
                    self._model = False
                    self._rehash_existing_chunks()
                else:
                    self._embeddings = loaded_embeddings
                    if self._embedding_mode is None:
                        # Older stores were built with sentence-transformers before modes were persisted.
                        self._embedding_mode = "semantic"
                    self._rebuild_index()
        if self._chunks and not manifest_path.exists():
            # Migrate older stores to the manifest format after a successful legacy load.
            self._save()
        if manifest_path.exists():
            self._loaded_manifest_sha = _sha256_file(manifest_path)
        logger.info("Loaded vector store with %d chunks", len(self._chunks))
