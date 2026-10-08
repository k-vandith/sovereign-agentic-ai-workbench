"""Tests for document processing."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.document import extract_text_from_file, chunk_text


def test_chunk_text_basic():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    assert all(len(c) <= 550 for c in chunks)


def test_chunk_text_short():
    assert chunk_text("hello") == ["hello"]
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_extract_txt(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("Industrial note about reactor safety.", encoding="utf-8")
    text = extract_text_from_file(f)
    assert "reactor" in text.lower()


def test_extract_unsupported(tmp_path: Path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text_from_file(f)


def test_extract_missing():
    with pytest.raises(FileNotFoundError):
        extract_text_from_file(Path("/nonexistent/file.txt"))
