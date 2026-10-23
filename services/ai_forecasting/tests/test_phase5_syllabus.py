"""
Phase 5 Unit Tests: Academic Syllabus PDF and Plain-Text Extraction & Parsing.
"""

from pathlib import Path
import pytest
from services.ai_forecasting.syllabus.extractor import extract_text_from_file, SyllabusExtractorError
from services.ai_forecasting.syllabus.parser import parse_syllabus_text


def test_extract_plain_text_syllabus():
    txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
    with open(txt_path, "rb") as f:
        content = f.read()

    text = extract_text_from_file(content, filename="sample_syllabus.txt")
    assert "Distributed Systems & Cloud Computing" in text

    parsed = parse_syllabus_text(text)
    assert parsed.course_name == "Distributed Systems & Cloud Computing"
    assert len(parsed.units) == 3
    assert parsed.units[0].unit_title == "Unit 1: Foundations of Distributed Systems"
    assert parsed.units[1].unit_title == "Unit 2: Containerization and Infrastructure"
    assert parsed.units[2].unit_title == "Unit 3: Service Communication and Protocols"

    # Verify skills detected
    assert "Linux" in parsed.extracted_skills
    assert "Docker" in parsed.extracted_skills
    assert "REST APIs" in parsed.extracted_skills
    # Missing skills should not be extracted
    assert "Kubernetes" not in parsed.extracted_skills
    assert "gRPC" not in parsed.extracted_skills


def test_extract_pdf_syllabus():
    pdf_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.pdf"
    with open(pdf_path, "rb") as f:
        content = f.read()

    text = extract_text_from_file(content, filename="sample_syllabus.pdf")
    assert "Distributed Systems" in text

    parsed = parse_syllabus_text(text)
    assert "Distributed Systems" in parsed.course_name
    assert len(parsed.units) >= 1
    assert "Linux" in parsed.extracted_skills or "Docker" in parsed.extracted_skills


def test_empty_file_error():
    with pytest.raises(SyllabusExtractorError, match="empty"):
        extract_text_from_file(b"", filename="empty.pdf")


def test_file_too_large_error():
    large_dummy = b"A" * (11 * 1024 * 1024)  # 11 MB
    with pytest.raises(SyllabusExtractorError, match="exceeds maximum permitted limit"):
        extract_text_from_file(large_dummy, filename="large.pdf")


def test_scanned_pdf_error():
    # A valid PDF structure with an empty text stream (simulating a scanned image without OCR)
    scanned_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 4 0 R >> endobj\n"
        b"4 0 obj << /Length 0 >> stream\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000214 00000 n \n"
        b"trailer << /Size 5 /Root 1 0 R >>\nstartxref\n264\n%%EOF\n"
    )
    with pytest.raises(SyllabusExtractorError, match="Scanned or image-only PDF detected"):
        extract_text_from_file(scanned_pdf, filename="scanned.pdf")


def test_missing_unit_headers_graceful_fallback():
    raw_text = """
    Advanced Python Web Engineering
    In this course students will learn web fundamentals.
    Topics include FastAPI development, database queries with SQL, and deployment with Docker.
    """
    parsed = parse_syllabus_text(raw_text)
    assert parsed.course_name == "Advanced Python Web Engineering"
    assert len(parsed.units) == 1
    assert parsed.units[0].unit_title == "General Course Content"
    assert "FastAPI" in parsed.extracted_skills
    assert "Docker" in parsed.extracted_skills
    assert "SQL" in parsed.extracted_skills
