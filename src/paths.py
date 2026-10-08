"""Safe path helpers. Reject traversal and odd upload names."""
from __future__ import annotations

from pathlib import Path


def safe_filename(name: str) -> str:
    """Return a single path segment safe to join under an upload directory."""
    if not name or not str(name).strip():
        raise ValueError("A filename is required.")
    base = Path(str(name).replace("\\", "/")).name
    if base in {"", ".", ".."}:
        raise ValueError("Invalid filename.")
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise ValueError("Invalid filename.")
    return cleaned[:180]
