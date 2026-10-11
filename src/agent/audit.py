"""Append-only, hash-chained audit log for prompts, tool calls, and answers."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_GENESIS_HASH = "0" * 64
_LEGACY_PREFIX_DOMAIN = b"sovereign-workbench-legacy-audit-prefix\0"


@contextmanager
def _exclusive_audit_lock(path: Path):
    """Serialize audit reads and writes across distinct app processes."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
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


class AuditLog:
    """Thread-safe JSONL audit log with SHA-256 tamper-evidence chaining.

    A separate head file stores the latest hash and entry count so that
    truncating the tail of the log is detectable (the in-file chain alone
    would still validate any prefix).
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("data/audit/audit.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.head_path = self.path.with_suffix(self.path.suffix + ".head")
        self._lock = threading.Lock()

    @staticmethod
    def _digest(entry_without_hash: dict[str, Any]) -> str:
        canonical = json.dumps(
            entry_without_hash,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _legacy_anchor(raw_prefix: bytes) -> str:
        return hashlib.sha256(_LEGACY_PREFIX_DOMAIN + raw_prefix).hexdigest()

    def _read_head(self) -> dict[str, Any] | None:
        if not self.head_path.exists():
            return None
        try:
            data = json.loads(self.head_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _write_head(self, head_hash: str | None, entry_count: int) -> None:
        payload = {
            "head_hash": head_hash,
            "entry_count": int(entry_count),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        temporary = self.head_path.with_suffix(self.head_path.suffix + ".tmp")
        try:
            temporary.write_bytes(encoded)
            temporary.replace(self.head_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _verify_bytes(self, raw: bytes) -> dict[str, Any]:
        previous_hash: str | None = None
        legacy_lines: list[bytes] = []
        checked_entries = 0
        legacy_entries = 0
        chain_started = False

        for line_number, line in enumerate(raw.splitlines(keepends=True), start=1):
            try:
                entry = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": f"Invalid JSON or UTF-8 on audit line {line_number}.",
                }
            if not isinstance(entry, dict):
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": f"Audit line {line_number} is not a JSON object.",
                }

            has_hash = "hash" in entry
            has_previous = "prev_hash" in entry
            if not has_hash and not has_previous:
                if chain_started:
                    return {
                        "valid": False,
                        "checked_entries": checked_entries,
                        "legacy_entries": legacy_entries,
                        "head_hash": previous_hash,
                        "error": f"Unchained audit entry found after the chain began (line {line_number}).",
                    }
                legacy_lines.append(line)
                legacy_entries += 1
                continue

            if not has_hash or not has_previous or not isinstance(entry.get("hash"), str):
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": f"Incomplete hash metadata on audit line {line_number}.",
                }

            if previous_hash is not None:
                expected_previous = previous_hash
            elif legacy_lines:
                expected_previous = self._legacy_anchor(b"".join(legacy_lines))
            else:
                expected_previous = _GENESIS_HASH

            if not hmac.compare_digest(str(entry["prev_hash"]), expected_previous):
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": f"Previous-hash mismatch on audit line {line_number}.",
                }

            unhashed_entry = {key: value for key, value in entry.items() if key != "hash"}
            expected_hash = self._digest(unhashed_entry)
            if not hmac.compare_digest(entry["hash"], expected_hash):
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": f"Entry hash mismatch on audit line {line_number}.",
                }

            previous_hash = entry["hash"]
            checked_entries += 1
            chain_started = True

        # External head anchor: detect tail truncation that would still form a
        # valid in-file prefix chain (or a fully emptied log).
        stored_head = self._read_head()
        if stored_head is not None:
            expected_count = stored_head.get("entry_count")
            expected_hash = stored_head.get("head_hash")
            if isinstance(expected_count, int) and expected_count != checked_entries:
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": (
                        f"Audit head mismatch: log has {checked_entries} chained entries "
                        f"but head records {expected_count} (possible tail truncation)."
                    ),
                }
            if (
                isinstance(expected_hash, str)
                and expected_hash
                and previous_hash is not None
                and not hmac.compare_digest(expected_hash, previous_hash)
            ):
                return {
                    "valid": False,
                    "checked_entries": checked_entries,
                    "legacy_entries": legacy_entries,
                    "head_hash": previous_hash,
                    "error": "Audit head hash does not match the last chained entry.",
                }

        if legacy_entries and not chain_started:
            message = (
                "Legacy entries have no individual hashes yet; they will be anchored "
                "when the first hash-chained event is appended."
            )
        elif legacy_entries:
            message = (
                "Hash chain is valid. The legacy prefix is anchored as a byte-for-byte "
                "digest but was not individually chain-verifiable before the upgrade."
            )
        else:
            message = "Hash chain is valid."

        return {
            "valid": True,
            "checked_entries": checked_entries,
            "legacy_entries": legacy_entries,
            "head_hash": previous_hash,
            "error": None,
            "message": message,
        }

    def record(self, event_type: str, **payload: Any) -> dict[str, Any]:
        """Append one event, refusing to extend an already-corrupt chain."""
        with self._lock, _exclusive_audit_lock(self.path):
            raw = self.path.read_bytes() if self.path.exists() else b""
            status = self._verify_bytes(raw)
            if not status["valid"]:
                raise RuntimeError(
                    "Audit log integrity check failed; refusing to append to a potentially tampered log."
                )

            if status["head_hash"]:
                previous_hash = status["head_hash"]
            elif raw:
                previous_hash = self._legacy_anchor(raw)
            else:
                previous_hash = _GENESIS_HASH

            entry: dict[str, Any] = {
                **payload,
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": event_type,
                "prev_hash": previous_hash,
            }
            entry["hash"] = self._digest(entry)
            encoded = (json.dumps(entry, ensure_ascii=False, default=str) + "\n").encode("utf-8")
            with self.path.open("ab") as stream:
                stream.write(encoded)
                stream.flush()
            new_count = int(status["checked_entries"]) + 1
            self._write_head(entry["hash"], new_count)
            return entry

    def verify(self) -> dict[str, Any]:
        """Verify every available chain link and report any legacy unchained prefix."""
        with self._lock, _exclusive_audit_lock(self.path):
            raw = self.path.read_bytes() if self.path.exists() else b""
            return self._verify_bytes(raw)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent valid JSON records while bounding memory use."""
        if limit <= 0:
            return []
        with self._lock, _exclusive_audit_lock(self.path):
            if not self.path.exists():
                return []
            with self.path.open("r", encoding="utf-8") as stream:
                lines = deque(stream, maxlen=limit)
        out: list[dict[str, Any]] = []
        for line in lines:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
        return out
