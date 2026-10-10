"""CPU semantic + BM25 hybrid retrieval with a dev-set calibrated abstention threshold."""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Iterable

import numpy as np

from .store import DocumentChunk, LocalVectorStore

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_CALIBRATION_VERSION = "hybrid-dev-v1"
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"\b([A-Z]{1,4})[-\s]?(\d{1,6}[A-Z]?)\b", re.IGNORECASE)
_TITLE_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "during",
    "for", "from", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or",
    "our", "please", "should", "so", "tell", "that", "the", "their", "then", "there",
    "this", "to", "us", "we", "what", "when", "where", "which", "who", "why", "with",
    "you", "your", "off", "won", "need", "take", "much", "long", "often", "every",
}
_TITLE_GENERIC_WORDS = {
    "assembly", "battery", "bearing", "boiler", "coolant", "compressor", "emergency",
    "eye", "fan", "filter", "gasket", "hot", "lubricant", "maximum", "minimum", "motor",
    "pump", "purge", "recipe", "reactor", "relay", "safety", "seal", "sensor", "tank",
    "valve", "water", "work", "highest", "lowest", "working", "pressure", "flow", "rate",
    "temperature", "inspection", "interval", "duration", "replacement", "capacity",
    "setting", "voltage", "speed", "clearance", "retention", "vibration", "sampling",
    "calibration", "allowed", "rated", "documented", "plant",
}

# Labelled dev examples are distinct from the held-out QA harness questions.
_DEV_DOCUMENTS = [
    ("dev_valves.txt", "Valve V-517 maximum working pressure is 34.5 bar. Valve V-518 is installed on the return line."),
    ("dev_pumps.txt", "Pump P-604 serves tank T-8 at 110 litres per minute. Pump P-605 is the standby unit."),
    ("dev_safety.txt", "During an ammonia leak, evacuate to assembly point C and notify the shift supervisor."),
    ("dev_recipe.txt", "Recipe RR-31 must be heated to 165 degrees Celsius for 27 minutes before cooling."),
    ("dev_fan.txt", "Fan F-62 has a rated operating speed of 1380 RPM."),
    ("dev_filter.txt", "Filter F-74 must be replaced after 52 operating hours."),
    ("dev_battery.txt", "Battery BT-9 has a minimum voltage of 24.1 volts."),
    ("dev_relay.txt", "Relay RY-719 trips at a pressure of 5.8 bar."),
]
_DEV_VALID = [
    ("how much pressure can valve V-518 handle", "dev_valves.txt"),
    ("which tank does pump P-604 feed", "dev_pumps.txt"),
    ("what should operators do during an ammonia leak", "dev_safety.txt"),
    ("how long must recipe RR-31 be heated", "dev_recipe.txt"),
    ("what speed is fan F-62 rated for", "dev_fan.txt"),
    ("when should filter F-74 be replaced", "dev_filter.txt"),
    ("what is the minimum voltage on battery BT-9", "dev_battery.txt"),
    ("at what pressure does relay RY-719 trip", "dev_relay.txt"),
]
_DEV_OUT_OF_SCOPE = [
    "who won the cricket world cup last season",
    "how do i bake a chocolate cake",
    "what is the salary of the plant manager",
    "summarize quarterly revenue projections",
    "what is today's exchange rate",
    "how do i tune an acoustic guitar",
    "write a poem about winter",
    "translate good morning into Spanish",
]


def _tokens(text: str) -> list[str]:
    result: list[str] = []
    for match in _TOKEN_RE.findall(str(text).lower()):
        if match not in _STOP_WORDS:
            result.append(match)
        parts = re.split(r"[-./]", match)
        if len(parts) > 1:
            result.extend(part for part in parts if part and part not in _STOP_WORDS)
    return result


def _bm25_scores(query: str, documents: list[str]) -> np.ndarray:
    """Dependency-free BM25 with a saturating normalization for hybrid scoring."""
    docs = [_tokens(text) for text in documents]
    n_docs = len(docs)
    if not n_docs:
        return np.zeros(0, dtype=np.float32)
    doc_terms = [Counter(tokens) for tokens in docs]
    lengths = [len(tokens) for tokens in docs]
    avg_len = (sum(lengths) / n_docs) if n_docs else 0.0
    document_frequency: Counter[str] = Counter()
    for terms in doc_terms:
        document_frequency.update(terms.keys())
    query_terms = set(_tokens(query))
    scores = np.zeros(n_docs, dtype=np.float32)
    if not query_terms:
        return scores
    k1, b = 1.5, 0.75
    for row, terms in enumerate(doc_terms):
        raw_score = 0.0
        doc_len = lengths[row]
        for term in query_terms:
            tf = terms.get(term, 0)
            if not tf:
                continue
            df = document_frequency[term]
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * (doc_len / max(avg_len, 1e-9)))
            raw_score += idf * ((tf * (k1 + 1.0)) / denominator)
        scores[row] = raw_score / (raw_score + 3.0) if raw_score > 0 else 0.0
    return scores


def calibrate_relevance_threshold(
    positive_scores: Iterable[float], negative_scores: Iterable[float]
) -> float:
    """Choose an abstention threshold by balanced accuracy on labelled dev scores."""
    positives = [float(score) for score in positive_scores]
    negatives = [float(score) for score in negative_scores]
    if not positives or not negatives:
        raise ValueError("Threshold calibration needs both relevant and out-of-scope dev scores.")
    values = sorted({0.0, 1.0, *positives, *negatives})
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2.0 for left, right in zip(values, values[1:]))
    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.28
    for threshold in candidates:
        true_positive_rate = sum(score > threshold for score in positives) / len(positives)
        true_negative_rate = sum(score <= threshold for score in negatives) / len(negatives)
        key = ((true_positive_rate + true_negative_rate) / 2.0, true_negative_rate, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return round(float(best_threshold), 4)


class HybridVectorStore(LocalVectorStore):
    """Vector store using FastEmbed/ONNX on CPU and BM25, with safe fallback."""

    def __init__(self, persist_dir, embedding_model_name: str = DEFAULT_MODEL) -> None:
        self._calibrated_threshold: float | None = None
        self._calibration_key: str | None = None
        self._model_error: str | None = None
        super().__init__(persist_dir=persist_dir, embedding_model_name=embedding_model_name or DEFAULT_MODEL)

    @property
    def calibrated_threshold(self) -> float | None:
        """Threshold cached from the bundled labelled development set for the active backend."""
        return self._calibrated_threshold

    def calibrate_threshold(self) -> float:
        """Explicitly fit/cache the threshold using independent labelled dev examples."""
        self._ensure_calibrated_threshold()
        assert self._calibrated_threshold is not None
        return self._calibrated_threshold

    def _backend_key(self) -> str:
        if self._embedding_mode == "semantic":
            return f"fastembed:{self.embedding_model_name}:{_CALIBRATION_VERSION}"
        return f"hashed-bm25:{_CALIBRATION_VERSION}"

    def _ensure_calibrated_threshold(self) -> None:
        if self._calibrated_threshold is not None:
            return
        self._get_model()
        cache_path = self.persist_dir / "retrieval_thresholds.json"
        cache: dict[str, float] = {}
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                cache = {str(key): float(value) for key, value in loaded.items()}
        except (OSError, ValueError, TypeError):
            cache = {}
        key = self._backend_key()
        if key in cache and 0.0 <= cache[key] <= 1.0:
            self._calibration_key = key
            self._calibrated_threshold = cache[key]
            return

        dev_chunks = [
            DocumentChunk(
                id=f"dev-{index}",
                text=text,
                source=source,
                metadata={"calibration": _CALIBRATION_VERSION},
            )
            for index, (source, text) in enumerate(_DEV_DOCUMENTS)
        ]
        dev_vectors = self._embed([chunk.text for chunk in dev_chunks])
        positive_scores: list[float] = []
        negative_scores: list[float] = []
        for question, expected_source in _DEV_VALID:
            scores = self._score_documents(question, dev_chunks, dev_vectors)
            expected = [scores[i] for i, chunk in enumerate(dev_chunks) if chunk.source == expected_source]
            if expected:
                positive_scores.append(max(expected))
        for question in _DEV_OUT_OF_SCOPE:
            scores = self._score_documents(question, dev_chunks, dev_vectors)
            negative_scores.append(max(scores, default=0.0))
        self._calibrated_threshold = calibrate_relevance_threshold(positive_scores, negative_scores)
        self._calibration_key = key
        cache[key] = self._calibrated_threshold
        try:
            temporary_path = cache_path.with_suffix(".tmp")
            temporary_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
            temporary_path.replace(cache_path)
        except OSError as exc:
            logger.debug("Could not persist calibrated threshold cache: %s", exc)

    def _get_model(self):
        if self._model is False:
            return None
        if self._model is not None:
            return self._model
        disabled = os.getenv("FASTEMBED_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}
        try:
            if disabled:
                raise ImportError("FastEmbed was disabled by FASTEMBED_DISABLE")
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.embedding_model_name or DEFAULT_MODEL,
                providers=["CPUExecutionProvider"],
            )
            self._embedding_mode = "semantic"
            self._semantic_status = "FastEmbed ONNX model active on CPU"
            self._model_error = None
            return self._model
        except Exception as exc:
            self._model = False
            self._model_error = type(exc).__name__
            if self._embedding_mode == "semantic" and self._chunks:
                self._rehash_existing_chunks()
            else:
                self._embedding_mode = "hashed"
            self._semantic_status = "FastEmbed unavailable; using hashed-text + BM25 fallback"
            logger.warning("FastEmbed unavailable; using hashed-text + BM25 retrieval (%s)", type(exc).__name__)
            return None

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        if model is None:
            self._embedding_mode = "hashed"
            return self._hashed_text_embeddings(texts)
        try:
            passages = [text if str(text).startswith("passage:") else f"passage: {text}" for text in texts]
            vectors = np.asarray(list(model.embed(passages, batch_size=64)), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(texts) or not np.isfinite(vectors).all():
                raise ValueError("FastEmbed returned invalid passage vectors")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.where(norms == 0.0, 1.0, norms)
        except Exception as exc:
            self._model_error = type(exc).__name__
            self._model = False
            if self._embedding_mode == "semantic" and self._chunks:
                self._rehash_existing_chunks()
            else:
                self._embedding_mode = "hashed"
            self._semantic_status = "FastEmbed unavailable; using hashed-text + BM25 fallback"
            logger.warning("FastEmbed embedding failed; using hashed-text + BM25 (%s)", type(exc).__name__)
            return self._hashed_text_embeddings(texts)

    def _embed_query(self, query: str) -> np.ndarray:
        model = self._get_model()
        if model is None:
            return self._hashed_text_embeddings([query])[0]
        try:
            # BGE retrieval prompts distinguish a query from a passage.
            vector = np.asarray(list(model.query_embed([f"query: {query}"])), dtype=np.float32)
            if vector.ndim != 2 or vector.shape[0] != 1 or not np.isfinite(vector).all():
                vector = np.asarray(list(model.embed([f"query: {query}"])), dtype=np.float32)
            result = vector[0]
            norm = float(np.linalg.norm(result))
            return result / norm if norm else result
        except AttributeError:
            try:
                vector = np.asarray(list(model.embed([f"query: {query}"])), dtype=np.float32)[0]
                norm = float(np.linalg.norm(vector))
                return vector / norm if norm else vector
            except Exception as exc:
                return self._switch_query_to_fallback(query, exc)
        except Exception as exc:
            return self._switch_query_to_fallback(query, exc)

    def _switch_query_to_fallback(self, query: str, exc: Exception) -> np.ndarray:
        logger.warning("FastEmbed query failed; using hashed-text + BM25 (%s)", type(exc).__name__)
        self._model = False
        self._model_error = type(exc).__name__
        self._embedding_mode = "hashed"
        self._semantic_status = "FastEmbed unavailable; using hashed-text + BM25 fallback"
        if self._chunks:
            self._rehash_existing_chunks()
        return self._hashed_text_embeddings([query])[0]

    @staticmethod
    def _reference_ids(text: str) -> set[str]:
        return {f"{match.group(1)}{match.group(2)}".upper() for match in _IDENTIFIER_RE.finditer(text)}

    def _question_has_missing_reference(self, question: str) -> bool:
        mentioned = self._reference_ids(question)
        if not mentioned:
            return False
        indexed = self._reference_ids(" ".join(chunk.text for chunk in self._chunks))
        return any(identifier not in indexed for identifier in mentioned)

    def _question_has_missing_named_entity(self, question: str) -> bool:
        corpus = " ".join(chunk.text for chunk in self._chunks).casefold()
        for match in _TITLE_WORD_RE.finditer(question):
            if match.start() == 0:
                continue
            word = match.group(0)
            if word.casefold() in _TITLE_GENERIC_WORDS:
                continue
            if word.casefold() not in corpus:
                return True
        return False

    def _score_documents(
        self,
        query: str,
        chunks: list[DocumentChunk],
        document_vectors: np.ndarray,
    ) -> list[float]:
        if not chunks:
            return []
        q_vector = self._embed_query(query)
        vectors = np.asarray(document_vectors, dtype=np.float32)
        denominator = np.linalg.norm(vectors, axis=1) * max(float(np.linalg.norm(q_vector)), 1e-12)
        denominator = np.where(denominator == 0.0, 1e-12, denominator)
        cosine = (vectors @ q_vector) / denominator
        dense_scores = np.clip((cosine - 0.12) / 0.68, 0.0, 1.0)
        lexical_scores = _bm25_scores(query, [chunk.text for chunk in chunks])
        combined = 0.62 * dense_scores + 0.38 * lexical_scores
        return [float(score) for score in combined]

    def _search_unlocked(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        if not self._chunks or self._embeddings is None:
            return []
        if self._question_has_missing_reference(query) or self._question_has_missing_named_entity(query):
            return []
        self._ensure_calibrated_threshold()
        scores = self._score_documents(query, self._chunks, self._embeddings)
        ranked_indices = np.argsort(np.asarray(scores))[::-1][: min(top_k, len(self._chunks))]
        return [(self._chunks[int(index)], scores[int(index)]) for index in ranked_indices]

    def embedding_status(self) -> str:
        """Return a truthful description of the active retrieval backend."""
        if self._embedding_mode == "semantic" and self._model not in (None, False):
            return "FastEmbed BGE-small semantic embeddings + BM25 (CPU/ONNX)"
        if self._model is False or self._embedding_mode == "hashed":
            return "Hashed-text + BM25 fallback (CPU; FastEmbed/model unavailable)"
        return "FastEmbed BGE-small semantic embeddings + BM25 (CPU/ONNX; lazy-loaded)"
