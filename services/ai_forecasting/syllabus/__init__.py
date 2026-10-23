"""
Syllabus document extraction and parsing module.
"""

from services.ai_forecasting.syllabus.extractor import extract_text_from_file, SyllabusExtractorError
from services.ai_forecasting.syllabus.parser import parse_syllabus_text

__all__ = ["extract_text_from_file", "parse_syllabus_text", "SyllabusExtractorError"]
