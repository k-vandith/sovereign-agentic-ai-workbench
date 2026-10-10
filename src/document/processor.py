"""Document ingestion and text extraction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)

SUPPORTED_TEXT = {".txt", ".md", ".csv", ".json", ".log"}
SUPPORTED_PDF = {".pdf"}
SUPPORTED_DOCX = {".docx"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def extract_text_from_file(path: Path) -> str:
    """Extract plain text from a supported file.

    Raises
    ------
    ValueError
        If the file type is unsupported or extraction fails.
    """
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        if suffix in SUPPORTED_TEXT:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix in SUPPORTED_PDF:
            return _extract_pdf(path)
        if suffix in SUPPORTED_DOCX:
            return _extract_docx(path)
        if suffix in SUPPORTED_IMAGE:
            return _extract_image(path)
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_DOCX | SUPPORTED_IMAGE}"
        )
    except Exception as exc:
        logger.exception("Failed to extract text from %s", path)
        raise ValueError(f"Failed to process {path.name}: {exc}") from exc


def _extract_image(path: Path) -> str:
    """Extract image text when optional Tesseract OCR is installed; otherwise register metadata."""
    from PIL import Image

    try:
        with Image.open(path) as image:
            width, height = image.size
            image_format = image.format or path.suffix.lstrip(".").upper()
            mode = image.mode
            try:
                import pytesseract
            except ImportError:
                return (
                    f"[Image file: {path.name}]\\nFormat: {image_format}, mode: {mode}, size: {width}x{height} px\\n"
                    "OCR is not enabled. Install the optional OCR extra and the Tesseract executable to index image text."
                )
            try:
                extracted = pytesseract.image_to_string(image).strip()
            except Exception as exc:
                logger.info("Optional OCR is unavailable for %s: %s", path.name, exc)
                extracted = ""
            if extracted:
                return (
                    f"[Image file: {path.name}]\\nFormat: {image_format}, size: {width}x{height} px\\n"
                    "OCR text extracted locally:\\n" + extracted
                )
            return (
                f"[Image file: {path.name}]\\nFormat: {image_format}, mode: {mode}, size: {width}x{height} px\\n"
                "No OCR text was available. The file is indexed as image metadata only."
            )
    except Exception as exc:
        raise ValueError(f"Could not open image file {path.name}. Check that it is a valid image.") from exc

def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"--- Page {i + 1} ---\n{text}")
    return "\n\n".join(parts) if parts else f"[PDF {path.name}: no extractable text]"


def _extract_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                idx = chunk.rfind(sep)
                if idx > chunk_size // 2:
                    end = start + idx + len(sep)
                    chunk = text[start:end]
                    break
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return [c for c in chunks if c]


def iter_supported_files(directory: Path) -> Iterator[Path]:
    """Yield supported files under a directory."""
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in (
            SUPPORTED_TEXT | SUPPORTED_PDF | SUPPORTED_DOCX | SUPPORTED_IMAGE
        ):
            yield p
