"""Safe in-memory upload handling for the local document library."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

from src.paths import safe_filename

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_BATCH_BYTES = 100 * 1024 * 1024
SUPPORTED_UPLOAD_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".md", ".csv", ".json", ".log",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp",
}


def _upload_bytes(upload: Any) -> bytes:
    getter = getattr(upload, "getvalue", None)
    if callable(getter):
        raw = getter()
    else:
        raw = getattr(upload, "data", b"")
    if not isinstance(raw, bytes):
        raw = bytes(raw)
    return raw


def ingest_uploaded_files(
    agent: Any,
    uploads: Iterable[Any],
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_batch_bytes: int = MAX_BATCH_BYTES,
) -> list[dict[str, Any]]:
    """Ingest several uploaded objects using a temporary folder that is always cleaned.

    Upload objects need a name and either getvalue() or a bytes-valued data
    attribute. Each result is independent: one unsupported file does not hide
    the status of the other files in the batch.
    """
    batch = list(uploads)
    if not batch:
        return []

    ready: list[tuple[str, str, bytes]] = []
    total_bytes = 0
    validation_errors: list[dict[str, Any]] = []
    for upload in batch:
        raw_name = str(getattr(upload, "name", "") or "")
        try:
            name = safe_filename(raw_name)
        except ValueError as exc:
            validation_errors.append({
                "file": raw_name or "Unnamed file",
                "status": "Not loaded",
                "chunks_added": 0,
                "message": str(exc),
            })
            continue

        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
            validation_errors.append({
                "file": name,
                "status": "Not loaded",
                "chunks_added": 0,
                "message": "Unsupported file type. Choose PDF, DOCX, TXT, MD, CSV, JSON, LOG, PNG, JPG, WEBP or BMP.",
            })
            continue

        try:
            raw = _upload_bytes(upload)
        except Exception:
            validation_errors.append({
                "file": name,
                "status": "Not loaded",
                "chunks_added": 0,
                "message": "The file bytes could not be read. Try selecting the file again.",
            })
            continue

        if not raw:
            validation_errors.append({
                "file": name,
                "status": "Not loaded",
                "chunks_added": 0,
                "message": "The file is empty.",
            })
            continue
        if len(raw) > max_file_bytes:
            validation_errors.append({
                "file": name,
                "status": "Not loaded",
                "chunks_added": 0,
                "message": f"This file exceeds the {max_file_bytes // (1024 * 1024)} MB per-file limit.",
            })
            continue
        total_bytes += len(raw)
        ready.append((name, suffix, raw))

    if total_bytes > max_batch_bytes:
        return validation_errors + [{
            "file": f"{len(ready)} selected file(s)",
            "status": "Not loaded",
            "chunks_added": 0,
            "message": f"The selected batch exceeds the {max_batch_bytes // (1024 * 1024)} MB total limit. Remove some files and retry.",
        }]

    results = list(validation_errors)
    with TemporaryDirectory(prefix="sovereign-workbench-upload-") as folder:
        used_names: set[str] = set()
        for name, _suffix, raw in ready:
            stem, suffix = Path(name).stem, Path(name).suffix
            candidate = name
            duplicate = 2
            while candidate.casefold() in used_names:
                candidate = f"{stem}_{duplicate}{suffix}"
                duplicate += 1
            used_names.add(candidate.casefold())
            path = Path(folder) / candidate
            try:
                path.write_bytes(raw)
                outcome = agent.ingest_file(path)
                results.append({
                    "file": name,
                    "status": "Indexed",
                    "chunks_added": int(outcome.get("chunks_added", 0)),
                    "total_chunks": int(outcome.get("total_chunks", 0)),
                    "message": "Available for local search.",
                })
            except (ValueError, FileNotFoundError, PermissionError) as exc:
                results.append({
                    "file": name,
                    "status": "Not loaded",
                    "chunks_added": 0,
                    "message": str(exc),
                })
            except Exception:
                results.append({
                    "file": name,
                    "status": "Not loaded",
                    "chunks_added": 0,
                    "message": "Processing failed unexpectedly. Try another file or inspect the local application log.",
                })
    return results
