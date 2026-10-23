"""
Syllabus document text extraction service.
Supports PDF and plain-text files with size validation and scanned PDF detection.
"""

import io
from pathlib import Path
from typing import Optional, Union
from pypdf import PdfReader
from services.ai_forecasting.config import settings


class SyllabusExtractorError(ValueError):
    """Custom error for syllabus validation or extraction failures."""
    pass


def extract_text_from_file(
    content: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """
    Extracts plain text from raw file bytes.
    Validates file size, format, and verifies that PDFs contain extractable text (not scanned images).
    """
    # 1. Size validation
    if len(content) == 0:
        raise SyllabusExtractorError("Uploaded file is empty (0 bytes).")

    if len(content) > settings.MAX_SYLLABUS_FILE_SIZE_BYTES:
        max_mb = settings.MAX_SYLLABUS_FILE_SIZE_BYTES / (1024 * 1024)
        raise SyllabusExtractorError(
            f"File size exceeds maximum permitted limit of {max_mb:.0f} MB."
        )

    # 2. Format determination
    fn = (filename or "").lower()
    ct = (content_type or "").lower()

    is_pdf = fn.endswith(".pdf") or "pdf" in ct or content.startswith(b"%PDF")
    is_text = fn.endswith((".txt", ".md", ".text")) or "text" in ct

    if not is_pdf and not is_text:
        # Default fallback: try utf-8 decode or check magic bytes
        if content.startswith(b"%PDF"):
            is_pdf = True
        else:
            is_text = True

    # 3. PDF Extraction
    if is_pdf:
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as e:
            raise SyllabusExtractorError(f"Corrupted or invalid PDF structure: {e}")

        if len(reader.pages) == 0:
            raise SyllabusExtractorError("PDF contains 0 pages.")

        extracted_pages = []
        for idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                extracted_pages.append(page_text.strip())
            except Exception as e:
                extracted_pages.append("")

        full_text = "\n\n".join(filter(None, extracted_pages)).strip()

        # Scanned PDF check: document has pages but virtually no extractable text characters
        if len(full_text) < 20:
            raise SyllabusExtractorError(
                "Scanned or image-only PDF detected (insufficient extractable text). "
                "Optical Character Recognition (OCR) is required to parse this document."
            )

        return full_text

    # 4. Plain Text Extraction
    try:
        text = content.decode("utf-8").strip()
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1").strip()
        except Exception as e:
            raise SyllabusExtractorError(f"Could not decode text file: {e}")

    if not text:
        raise SyllabusExtractorError("Text document contains no extractable content.")

    return text
