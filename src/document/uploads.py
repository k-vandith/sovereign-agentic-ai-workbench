"""Validated, disk-staged upload handling for the local document library."""
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


def _declared_upload_size(upload: Any) -> int | None:
    """Read trustworthy size metadata without copying the upload body into memory."""
    size = getattr(upload, "size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return size
    getter = getattr(upload, "getbuffer", None)
    if callable(getter):
        try:
            return len(getter())
        except Exception:
            return None
    return None


def _stage_upload(upload: Any, destination: Path) -> int:
    """Write one upload to disk without retaining a second in-memory copy when possible."""
    getter = getattr(upload, "getbuffer", None)
    if callable(getter):
        view = getter()
        with destination.open("wb") as stream:
            stream.write(view)
        return len(view)

    value_getter = getattr(upload, "getvalue", None)
    raw = value_getter() if callable(value_getter) else getattr(upload, "data", b"")
    if not isinstance(raw, bytes):
        raw = bytes(raw)
    with destination.open("wb") as stream:
        stream.write(raw)
    return len(raw)


def ingest_uploaded_files(
    agent: Any,
    uploads: Iterable[Any],
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_batch_bytes: int = MAX_BATCH_BYTES,
) -> list[dict[str, Any]]:
    """Validate and stage a batch before indexing any file.

    Uploads with size metadata are rejected before their contents are copied or staged.
    Validated data is then staged one file at a time in a temporary directory, so this
    function does not retain a second in-memory copy of the full batch.
    """
    batch = list(uploads)
    if not batch:
        return []

    candidates: list[tuple[Any, str, str, int | None]] = []
    validation_errors: list[dict[str, Any]] = []
    declared_total = 0

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

        size = _declared_upload_size(upload)
        if size == 0:
            validation_errors.append({
                "file": name,
                "status": "Not loaded",
                "chunks_added": 0,
                "message": "The file is empty.",
            })
            continue
        if size is not None and size > max_file_bytes:
            validation_errors.append({
                "file": name,
                "status": "Not loaded",
                "chunks_added": 0,
                "message": f"This file exceeds the {max_file_bytes // (1024 * 1024)} MB per-file limit.",
            })
            continue

        if size is not None:
            declared_total += size
        candidates.append((upload, name, suffix, size))

    if declared_total > max_batch_bytes:
        return validation_errors + [{
            "file": f"{len(batch)} selected file(s)",
            "status": "Not loaded",
            "chunks_added": 0,
            "message": f"The selected batch exceeds the {max_batch_bytes // (1024 * 1024)} MB total limit. Remove some files and retry.",
        }]

    results = list(validation_errors)
    with TemporaryDirectory(prefix="sovereign-workbench-upload-") as folder:
        used_names: set[str] = set()
        staged: list[tuple[str, Path]] = []
        actual_total = 0

        for upload, name, _suffix, _declared_size in candidates:
            stem, suffix = Path(name).stem, Path(name).suffix
            candidate = name
            duplicate = 2
            while candidate.casefold() in used_names:
                candidate = f"{stem}_{duplicate}{suffix}"
                duplicate += 1
            used_names.add(candidate.casefold())
            path = Path(folder) / candidate

            try:
                size = _stage_upload(upload, path)
            except Exception:
                path.unlink(missing_ok=True)
                results.append({
                    "file": name,
                    "status": "Not loaded",
                    "chunks_added": 0,
                    "message": "The file bytes could not be read. Try selecting the file again.",
                })
                continue

            if size == 0:
                path.unlink(missing_ok=True)
                results.append({
                    "file": name,
                    "status": "Not loaded",
                    "chunks_added": 0,
                    "message": "The file is empty.",
                })
                continue
            if size > max_file_bytes:
                path.unlink(missing_ok=True)
                results.append({
                    "file": name,
                    "status": "Not loaded",
                    "chunks_added": 0,
                    "message": f"This file exceeds the {max_file_bytes // (1024 * 1024)} MB per-file limit.",
                })
                continue

            actual_total += size
            if actual_total > max_batch_bytes:
                return results + [{
                    "file": f"{len(batch)} selected file(s)",
                    "status": "Not loaded",
                    "chunks_added": 0,
                    "message": f"The selected batch exceeds the {max_batch_bytes // (1024 * 1024)} MB total limit. Remove some files and retry.",
                }]
            staged.append((name, path))

        # No document enters the index until the whole selected batch is within limits.
        for name, path in staged:
            try:
                outcome = agent.ingest_file(path)
                chunks_added = int(outcome.get("chunks_added", 0))
                results.append({
                    "file": name,
                    "status": "Indexed" if chunks_added else "No searchable text",
                    "chunks_added": chunks_added,
                    "total_chunks": int(outcome.get("total_chunks", 0)),
                    "message": (
                        "Available for local search."
                        if chunks_added
                        else "No searchable text was extracted; this file is not available for retrieval."
                    ),
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
