"""
test_contract_formatter.py — Phase 6 Test Checkpoint
=====================================================
Tests contract_formatter.py in full isolation.

Coverage:
  1. Valid Contract A payload formatting and schema adherence.
  2. Strict schema validator: missing keys, invalid types, extra keys.
  3. `job_id` normalization: strings, integers, whitespace trimming.
  4. Date normalization: date, datetime, ISO strings, fallback to UTC today.
  5. Calendar date validity (e.g. rejection of February 30).
  6. Skill list deduplication, sorting, and type enforcement.
  7. `format_from_raw_skills()` end-to-end integration with normalizer.
  8. Batch formatting (`format_contract_a_batch`).
  9. JSON serialization (`to_contract_json`) and disk export (`save_contract_json`).

Run with: python -m unittest services/nlp_engine/tests/test_contract_formatter.py -v
"""

from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp_engine.output.contract_formatter import (
    CONTRACT_A_REQUIRED_KEYS,
    ContractValidationError,
    format_contract_a,
    format_contract_a_batch,
    format_from_raw_skills,
    save_contract_json,
    to_contract_json,
    validate_contract_a,
)


# ---------------------------------------------------------------------------
# Test Suite 1: Schema Validation
# ---------------------------------------------------------------------------

class TestValidateContractA(unittest.TestCase):
    """Tests the validate_contract_a function against valid and invalid payloads."""

    def test_valid_payload_passes(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Docker", "FastAPI", "Python"],
            "posting_date": "2026-10-01",
        }
        self.assertTrue(validate_contract_a(payload))

    def test_valid_payload_with_empty_skills_passes(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": [],
            "posting_date": "2026-10-01",
        }
        self.assertTrue(validate_contract_a(payload))

    def test_missing_job_id_raises(self):
        payload = {
            "extracted_skills": ["Python"],
            "posting_date": "2026-10-01",
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_contract_a(payload)
        self.assertIn("job_id", str(ctx.exception))

    def test_missing_extracted_skills_raises(self):
        payload = {
            "job_id": "12345",
            "posting_date": "2026-10-01",
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_missing_posting_date_raises(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Python"],
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_extra_keys_in_strict_mode_raises(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Python"],
            "posting_date": "2026-10-01",
            "unexpected_field": "bonus",
        }
        with self.assertRaises(ContractValidationError) as ctx:
            validate_contract_a(payload, strict=True)
        self.assertIn("unexpected_field", str(ctx.exception))

    def test_extra_keys_allowed_when_not_strict(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Python"],
            "posting_date": "2026-10-01",
            "unexpected_field": "bonus",
        }
        self.assertTrue(validate_contract_a(payload, strict=False))

    def test_empty_job_id_raises(self):
        payload = {
            "job_id": "   ",
            "extracted_skills": ["Python"],
            "posting_date": "2026-10-01",
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_non_string_job_id_raises(self):
        payload = {
            "job_id": 12345,  # int not allowed directly in validator (formatter converts it)
            "extracted_skills": ["Python"],
            "posting_date": "2026-10-01",
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_invalid_date_format_raises(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Python"],
            "posting_date": "01-10-2026",  # DD-MM-YYYY not allowed
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_invalid_calendar_date_raises(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Python"],
            "posting_date": "2026-02-30",  # Feb 30 does not exist
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_non_list_extracted_skills_raises(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": "Python",  # String instead of list
            "posting_date": "2026-10-01",
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_non_string_element_in_skills_raises(self):
        payload = {
            "job_id": "12345",
            "extracted_skills": ["Python", 123],
            "posting_date": "2026-10-01",
        }
        with self.assertRaises(ContractValidationError):
            validate_contract_a(payload)

    def test_non_dict_payload_raises(self):
        with self.assertRaises(ContractValidationError):
            validate_contract_a(["not", "a", "dict"])


# ---------------------------------------------------------------------------
# Test Suite 2: Single-Job Formatter
# ---------------------------------------------------------------------------

class TestFormatContractA(unittest.TestCase):
    """Tests format_contract_a construction and normalization."""

    def test_format_with_valid_string_job_id(self):
        result = format_contract_a(
            job_id="job_999",
            extracted_skills=["Python", "FastAPI"],
            posting_date="2026-10-01",
        )
        self.assertEqual(result["job_id"], "job_999")
        self.assertEqual(result["extracted_skills"], ["FastAPI", "Python"])
        self.assertEqual(result["posting_date"], "2026-10-01")

    def test_format_converts_integer_job_id_to_string(self):
        result = format_contract_a(
            job_id=12345,
            extracted_skills=["Python"],
            posting_date="2026-10-01",
        )
        self.assertEqual(result["job_id"], "12345")
        self.assertIsInstance(result["job_id"], str)

    def test_format_trims_job_id_whitespace(self):
        result = format_contract_a(
            job_id="   job_42   ",
            extracted_skills=["Docker"],
            posting_date="2026-10-01",
        )
        self.assertEqual(result["job_id"], "job_42")

    def test_format_accepts_date_object(self):
        d = date(2026, 10, 15)
        result = format_contract_a(
            job_id="1",
            extracted_skills=["Docker"],
            posting_date=d,
        )
        self.assertEqual(result["posting_date"], "2026-10-15")

    def test_format_accepts_datetime_object(self):
        dt = datetime(2026, 10, 15, 14, 30, 0)
        result = format_contract_a(
            job_id="1",
            extracted_skills=["Docker"],
            posting_date=dt,
        )
        self.assertEqual(result["posting_date"], "2026-10-15")

    def test_format_normalizes_alternative_date_string(self):
        result = format_contract_a(
            job_id="1",
            extracted_skills=["Docker"],
            posting_date="2026/10/15",
        )
        self.assertEqual(result["posting_date"], "2026-10-15")

    def test_format_none_date_defaults_to_today(self):
        result = format_contract_a(
            job_id="1",
            extracted_skills=["Docker"],
            posting_date=None,
        )
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.assertEqual(result["posting_date"], today_str)

    def test_format_deduplicates_and_sorts_skills(self):
        result = format_contract_a(
            job_id="1",
            extracted_skills=["Python", "Docker", "Python", "FastAPI"],
            posting_date="2026-10-01",
            sort_skills=True,
        )
        self.assertEqual(result["extracted_skills"], ["Docker", "FastAPI", "Python"])

    def test_format_none_job_id_raises(self):
        with self.assertRaises(ContractValidationError):
            format_contract_a(job_id=None, extracted_skills=["Python"])

    def test_format_empty_job_id_raises(self):
        with self.assertRaises(ContractValidationError):
            format_contract_a(job_id="", extracted_skills=["Python"])


# ---------------------------------------------------------------------------
# Test Suite 3: End-to-End Formatter with Skill Normalizer
# ---------------------------------------------------------------------------

class TestFormatFromRawSkills(unittest.TestCase):
    """Tests format_from_raw_skills combining normalizer + formatter."""

    def test_format_from_raw_skill_strings(self):
        raw = ["React.js", "python3", "ReactJS", "fast-api"]
        result = format_from_raw_skills(
            job_id="jd-001",
            raw_skills_or_hits=raw,
            posting_date="2026-10-01",
        )
        self.assertEqual(result["job_id"], "jd-001")
        # All aliases resolved to canonical names and deduplicated
        self.assertEqual(result["extracted_skills"], ["FastAPI", "Python", "React"])
        self.assertEqual(result["posting_date"], "2026-10-01")

    def test_format_from_ner_hit_dicts(self):
        hits = [
            {"text": "React.js", "taxonomy_id": "t-101", "start_char": 0, "end_char": 8},
            {"text": "Python", "taxonomy_id": "t-001", "start_char": 10, "end_char": 16},
        ]
        result = format_from_raw_skills(
            job_id="jd-002",
            raw_skills_or_hits=hits,
            posting_date="2026-10-01",
        )
        self.assertEqual(result["extracted_skills"], ["Python", "React"])


# ---------------------------------------------------------------------------
# Test Suite 4: Batch Formatting
# ---------------------------------------------------------------------------

class TestBatchFormatting(unittest.TestCase):
    """Tests format_contract_a_batch."""

    def test_batch_formatting_multiple_records(self):
        records = [
            {"job_id": 101, "extracted_skills": ["Python", "Docker"], "posting_date": "2026-10-01"},
            {"job_id": 102, "skills": ["React", "TypeScript"], "date": "2026-10-02"},
        ]
        batch_out = format_contract_a_batch(records)
        self.assertEqual(len(batch_out), 2)
        self.assertEqual(batch_out[0]["job_id"], "101")
        self.assertEqual(batch_out[0]["extracted_skills"], ["Docker", "Python"])
        self.assertEqual(batch_out[1]["job_id"], "102")
        self.assertEqual(batch_out[1]["extracted_skills"], ["React", "TypeScript"])

    def test_batch_formatting_empty_list(self):
        self.assertEqual(format_contract_a_batch([]), [])


# ---------------------------------------------------------------------------
# Test Suite 5: JSON Export & Disk Serialization
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):
    """Tests to_contract_json and save_contract_json."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.record = {
            "job_id": "12345",
            "extracted_skills": ["Docker", "Python"],
            "posting_date": "2026-10-01",
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_to_contract_json(self):
        json_str = to_contract_json(self.record)
        parsed = json.loads(json_str)
        self.assertEqual(parsed, self.record)

    def test_save_contract_json_file(self):
        out_path = Path(self.tmp_dir) / "output" / "contract_a.json"
        saved = save_contract_json(self.record, out_path)
        self.assertTrue(saved.exists())

        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data, self.record)


if __name__ == "__main__":
    unittest.main()
