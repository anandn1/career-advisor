"""Extracts plain text from uploaded resume files."""

import logging
from pathlib import Path

import pdfplumber
from docx import Document
from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt")


class UnsupportedDocumentError(ValueError):
    pass


def _pdf_text(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if text.strip():
            return text
    except Exception as error:  # pdfplumber raises many unrelated types on bad PDFs
        logger.warning("pdfplumber failed on %s (%s); falling back to pypdf", path, error)
    return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)


def _docx_text(path: Path) -> str:
    return "\n".join(paragraph.text for paragraph in Document(str(path)).paragraphs)


def read_document_text(path: Path) -> str:
    """Return the text of a .pdf, .docx or .txt file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(path)
    if suffix == ".docx":
        return _docx_text(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    raise UnsupportedDocumentError(
        f"Unsupported file type {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}"
    )
