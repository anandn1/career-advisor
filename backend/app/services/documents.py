"""Extracts plain text from uploaded resume files."""

import logging
from pathlib import Path

import pdfplumber
from docx import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".pdf", ".docx", ".txt")


class DocumentError(ValueError):
    """The file cannot be turned into text."""


class UnsupportedDocumentError(DocumentError):
    pass


class UnreadableDocumentError(DocumentError):
    pass


def _pdf_text(path: Path) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        if text.strip():
            return text
    except Exception as error:  # pdfplumber raises many unrelated types on bad PDFs
        logger.warning("pdfplumber failed on %s (%s); falling back to pypdf", path, error)
    try:
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    except (PyPdfError, OSError, ValueError) as error:
        raise UnreadableDocumentError(f"Could not read PDF {path.name}: {error}") from error


def _docx_text(path: Path) -> str:
    """Paragraphs, then table cells (resumes often keep skills in tables)."""
    try:
        document = Document(str(path))
    except Exception as error:  # python-docx raises zipfile/KeyError/lxml errors on bad files
        raise UnreadableDocumentError(f"Could not read DOCX {path.name}: {error}") from error
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    cells = [
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
        if cell.text.strip()
    ]
    return "\n".join(paragraphs + cells)


def read_document_text(path: Path) -> str:
    """Return the text of a .pdf, .docx or .txt file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _pdf_text(path)
    if suffix == ".docx":
        return _docx_text(path)
    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise UnreadableDocumentError(f"Could not read {path.name}: {error}") from error
    raise UnsupportedDocumentError(
        f"Unsupported file type {suffix!r}; use one of {', '.join(SUPPORTED_SUFFIXES)}"
    )
