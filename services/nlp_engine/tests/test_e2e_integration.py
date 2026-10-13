"""
test_e2e_integration.py — Phase 7 End-to-End Integration Test Suite
===================================================================
Tests the entire NLP Engine pipeline from end to end using realistic
mock job descriptions from `mock_job_postings.json`.

Pipeline flow verified:
  Raw Job HTML / Text
          │
          ▼  [pipeline/text_cleaner.py]
    Clean Plain Text
          │
          ▼  [pipeline/ner_pipeline.py]
    SkillHit Mentions
          │
          ▼  [normalization/skill_normalizer.py]
    Canonical Skill Names
          │
     ┌────┴────────────────────────┐
     ▼                             ▼
[output/contract_formatter.py]  [cooccurrence/cooccurrence_matrix.py]
     │                             │
     ▼                             ▼
Contract A Output JSON         Symmetric Co-occurrence Matrix
(Contract with Member 1 & 3)   (Contract with Member 3 & 4)

Run with: python -m unittest services/nlp_engine/tests/test_e2e_integration.py -v
"""

import json
from pathlib import Path
import sys
import unittest

import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp_engine.cooccurrence.cooccurrence_matrix import CooccurrenceMatrix
from services.nlp_engine.engine import NLPEngine
from services.nlp_engine.output.contract_formatter import validate_contract_a

MOCK_POSTINGS_FILE = Path(__file__).resolve().parent / "mock_job_postings.json"


# ---------------------------------------------------------------------------
# Test Suite: End-to-End Integration
# ---------------------------------------------------------------------------

class TestEndToEndPipeline(unittest.TestCase):
    """Full integration tests over realistic mock job descriptions."""

    @classmethod
    def setUpClass(cls):
        """Loads mock postings and initialises the engine once for speed."""
        with open(MOCK_POSTINGS_FILE, encoding="utf-8") as f:
            cls.mock_jobs = json.load(f)

        # Uses spacy.blank("en") so it runs completely offline without model download
        nlp = spacy.blank("en")
        cls.engine = NLPEngine(nlp=nlp)

        # Execute full batch
        cls.records, cls.matrix = cls.engine.process_batch(
            cls.mock_jobs,
            build_cooccurrence=True,
        )

    def test_mock_jobs_loaded_successfully(self):
        """Asserts mock data has 12 diverse job postings."""
        self.assertEqual(len(self.mock_jobs), 12)

    def test_all_output_records_strictly_satisfy_contract_a(self):
        """Every single processed job must validate against Contract A schema."""
        self.assertEqual(len(self.records), len(self.mock_jobs))

        for record in self.records:
            # Must pass strict validation (no extra keys, correct types)
            self.assertTrue(validate_contract_a(record, strict=True))

            # Required Contract A fields
            self.assertIn("job_id", record)
            self.assertIn("extracted_skills", record)
            self.assertIn("posting_date", record)

            self.assertIsInstance(record["job_id"], str)
            self.assertIsInstance(record["extracted_skills"], list)
            self.assertIsInstance(record["posting_date"], str)

    def test_job_id_and_posting_dates_match_inputs(self):
        """Output records must retain correct job_id and date metadata."""
        for orig, output in zip(self.mock_jobs, self.records):
            self.assertEqual(output["job_id"], orig["job_id"])
            self.assertEqual(output["posting_date"], orig["posting_date"])

    def test_backend_python_job_skill_extraction(self):
        """Verifies accuracy on Job 101 (Python Backend)."""
        job = self.records[0]
        expected = self.mock_jobs[0]["expected_skills"]
        for skill in expected:
            self.assertIn(
                skill,
                job["extracted_skills"],
                f"Expected skill '{skill}' missing from Job 101 output",
            )

    def test_frontend_react_job_skill_extraction(self):
        """Verifies accuracy on Job 102 (React Frontend)."""
        job = self.records[1]
        expected = self.mock_jobs[1]["expected_skills"]
        for skill in expected:
            self.assertIn(
                skill,
                job["extracted_skills"],
                f"Expected skill '{skill}' missing from Job 102 output",
            )

    def test_devops_cloud_job_skill_extraction(self):
        """Verifies accuracy on Job 103 (DevOps / AWS)."""
        job = self.records[2]
        expected = self.mock_jobs[2]["expected_skills"]
        for skill in expected:
            self.assertIn(
                skill,
                job["extracted_skills"],
                f"Expected skill '{skill}' missing from Job 103 output",
            )

    def test_ai_agentic_job_skill_extraction(self):
        """Verifies accuracy on Job 106 (LangChain / Agentic AI)."""
        job = self.records[5]
        expected = self.mock_jobs[5]["expected_skills"]
        for skill in expected:
            self.assertIn(
                skill,
                job["extracted_skills"],
                f"Expected skill '{skill}' missing from Job 106 output",
            )

    def test_cooccurrence_matrix_properties(self):
        """Verifies the generated co-occurrence matrix is symmetric and zero-diagonal."""
        self.assertIsNotNone(self.matrix)
        df = self.matrix.matrix

        # Must have tracked skills
        self.assertGreater(len(df.index), 10)

        # Symmetry test
        for s1 in df.index:
            for s2 in df.columns:
                self.assertEqual(df.loc[s1, s2], df.loc[s2, s1])

        # Zero diagonal test
        for s in df.index:
            self.assertEqual(df.loc[s, s], 0)

    def test_cooccurrence_known_tech_stack_pairings(self):
        """Verifies that common companion skills have positive co-occurrence counts."""
        # Python and FastAPI co-occur in Job 101 and Job 106
        self.assertGreaterEqual(self.matrix.get_count("Python", "FastAPI"), 2)

        # React and Next.js co-occur in Job 102
        self.assertGreaterEqual(self.matrix.get_count("React", "Next.js"), 1)

        # LangChain and LangGraph co-occur in Job 106
        self.assertGreaterEqual(self.matrix.get_count("LangChain", "LangGraph"), 1)

        # Docker and Kubernetes co-occur in Job 101 and Job 108
        self.assertGreaterEqual(self.matrix.get_count("Docker", "Kubernetes"), 2)

    def test_network_edges_export_for_member_4(self):
        """Verifies network edge generator produces valid React Flow edge payloads."""
        edges = self.matrix.to_edges()
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 5)

        for edge in edges:
            self.assertIn("source", edge)
            self.assertIn("target", edge)
            self.assertIn("weight", edge)
            self.assertGreater(edge["weight"], 0)

    def test_single_job_process_workflow(self):
        """Verifies engine.process_job() produces valid Contract A record."""
        raw = "<p>Looking for a <b>Python</b> engineer with <b>FastAPI</b> & <b>Docker</b> experience.</p>"
        result = self.engine.process_job(
            job_id="single-999",
            raw_text=raw,
            posting_date="2026-10-15",
        )
        self.assertTrue(validate_contract_a(result, strict=True))
        self.assertEqual(result["job_id"], "single-999")
        self.assertEqual(result["posting_date"], "2026-10-15")
        self.assertEqual(result["extracted_skills"], ["Docker", "FastAPI", "Python"])

    def test_empty_raw_text_produces_empty_skills_list(self):
        """Job description with no text produces empty skills list without crashing."""
        result = self.engine.process_job(
            job_id="empty-1",
            raw_text="",
            posting_date="2026-10-01",
        )
        self.assertTrue(validate_contract_a(result, strict=True))
        self.assertEqual(result["extracted_skills"], [])


if __name__ == "__main__":
    unittest.main()
