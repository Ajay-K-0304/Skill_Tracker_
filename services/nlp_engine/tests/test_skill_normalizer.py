"""
test_skill_normalizer.py — Phase 4 Test Checkpoint
===================================================
Tests skill_normalizer.py in full isolation and with disk taxonomy.

Coverage:
  1. Isolated testing with injected mock taxonomy (fast, no disk dependency).
  2. Canonical resolution (case-insensitive, whitespace handling).
  3. Alias resolution to canonical name.
  4. Handling of unknown terms and malformed/empty inputs.
  5. Resolution of SkillHit dicts from NERPipeline (taxonomy_id matching).
  6. Confidence scoring (1.0 for rule/alias, <1.0 for fuzzy).
  7. Optional fuzzy matching fallback with difflib.
  8. Per-document deduplication by canonical name.
  9. `to_canonical_list()` output format for Contract A JSON.
  10. Batch normalization (`normalize_batch`, `batch_to_canonical_lists`).
  11. Real taxonomy integration with `skill_taxonomy.json`.

Run with: python -m unittest services/nlp_engine/tests/test_skill_normalizer.py -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp_engine.normalization.skill_normalizer import (
    SkillNormalizer,
    get_default_normalizer,
    normalize,
    normalize_list,
    to_canonical_list,
)

# ---------------------------------------------------------------------------
# Shared Mock Taxonomy for Isolated Tests
# ---------------------------------------------------------------------------

MOCK_TAXONOMY = [
    {
        "id": "t-001",
        "canonical_name": "Python",
        "category": "Language",
        "aliases": ["python3", "Python 3", "py"],
    },
    {
        "id": "t-101",
        "canonical_name": "React",
        "category": "Frontend",
        "aliases": ["React.js", "ReactJS", "react-js", "React JS"],
    },
    {
        "id": "t-201",
        "canonical_name": "FastAPI",
        "category": "Backend",
        "aliases": ["fast-api", "Fast API"],
    },
    {
        "id": "t-401",
        "canonical_name": "Docker",
        "category": "DevOps",
        "aliases": ["docker-compose", "Docker Compose", "Dockerfile"],
    },
    {
        "id": "t-005",
        "canonical_name": "C++",
        "category": "Language",
        "aliases": ["cpp", "C Plus Plus"],
    },
    {
        "id": "t-006",
        "canonical_name": "C#",
        "category": "Language",
        "aliases": ["csharp", "C Sharp"],
    },
]


def _make_normalizer() -> SkillNormalizer:
    return SkillNormalizer(taxonomy_entries=MOCK_TAXONOMY, fuzzy_threshold=0.80)


# ---------------------------------------------------------------------------
# Test Suite 1: Canonical & Exact Alias Resolution (Isolated)
# ---------------------------------------------------------------------------

class TestSkillNormalizerResolution(unittest.TestCase):
    """Tests exact matching and alias resolution."""

    def setUp(self):
        self.normalizer = _make_normalizer()

    def test_canonical_name_exact(self):
        result = self.normalizer.normalize("Python")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Python")
        self.assertEqual(result["taxonomy_id"], "t-001")
        self.assertEqual(result["category"], "Language")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["method"], "exact_alias")

    def test_canonical_name_lowercase(self):
        result = self.normalizer.normalize("python")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Python")

    def test_canonical_name_uppercase(self):
        result = self.normalizer.normalize("PYTHON")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Python")

    def test_canonical_name_with_whitespace(self):
        result = self.normalizer.normalize("   React   ")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "React")

    def test_alias_react_dot_js(self):
        result = self.normalizer.normalize("React.js")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "React")
        self.assertEqual(result["taxonomy_id"], "t-101")
        self.assertEqual(result["category"], "Frontend")

    def test_alias_reactjs(self):
        result = self.normalizer.normalize("reactjs")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "React")

    def test_alias_python3(self):
        result = self.normalizer.normalize("python3")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Python")

    def test_alias_fast_api_hyphen(self):
        result = self.normalizer.normalize("fast-api")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "FastAPI")

    def test_alias_docker_compose(self):
        result = self.normalizer.normalize("docker-compose")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Docker")

    def test_cpp_punctuation(self):
        result = self.normalizer.normalize("C++")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "C++")

    def test_cpp_alias(self):
        result = self.normalizer.normalize("cpp")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "C++")

    def test_csharp_alias(self):
        result = self.normalizer.normalize("csharp")
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "C#")


# ---------------------------------------------------------------------------
# Test Suite 2: SkillHit Dict Resolution (From NER Pipeline)
# ---------------------------------------------------------------------------

class TestSkillHitDictResolution(unittest.TestCase):
    """Tests resolution when input is a SkillHit dict produced by NERPipeline."""

    def setUp(self):
        self.normalizer = _make_normalizer()

    def test_resolve_by_valid_taxonomy_id(self):
        hit = {
            "text": "React.js",
            "label": "SKILL",
            "start_char": 10,
            "end_char": 18,
            "taxonomy_id": "t-101",
        }
        result = self.normalizer.normalize(hit)
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "React")
        self.assertEqual(result["taxonomy_id"], "t-101")
        self.assertEqual(result["category"], "Frontend")
        self.assertEqual(result["confidence"], 1.0)
        self.assertEqual(result["method"], "taxonomy_id")

    def test_resolve_by_text_when_taxonomy_id_unknown(self):
        hit = {
            "text": "python3",
            "label": "SKILL",
            "taxonomy_id": "nonexistent-id",
        }
        result = self.normalizer.normalize(hit)
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Python")
        self.assertEqual(result["taxonomy_id"], "t-001")


# ---------------------------------------------------------------------------
# Test Suite 3: Unknown & Invalid Inputs
# ---------------------------------------------------------------------------

class TestUnknownAndEdgeCaseInputs(unittest.TestCase):
    """Tests handling of unknown terms, empty strings, None, and edge cases."""

    def setUp(self):
        self.normalizer = _make_normalizer()

    def test_unknown_skill_returns_none(self):
        result = self.normalizer.normalize("TotallyMadeUpSkill999")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        self.assertIsNone(self.normalizer.normalize(""))

    def test_whitespace_string_returns_none(self):
        self.assertIsNone(self.normalizer.normalize("    "))

    def test_none_input_returns_none(self):
        self.assertIsNone(self.normalizer.normalize(None))

    def test_empty_dict_returns_none(self):
        self.assertIsNone(self.normalizer.normalize({}))

    def test_dict_with_whitespace_text_returns_none(self):
        self.assertIsNone(self.normalizer.normalize({"text": "   "}))

    def test_integer_input_handled_gracefully(self):
        self.assertIsNone(self.normalizer.normalize(12345))


# ---------------------------------------------------------------------------
# Test Suite 4: Fuzzy Fallback Matching
# ---------------------------------------------------------------------------

class TestFuzzyMatching(unittest.TestCase):
    """Tests standard-library difflib fuzzy matching fallback."""

    def setUp(self):
        self.normalizer = _make_normalizer()

    def test_fuzzy_disabled_by_default(self):
        # With allow_fuzzy=False, typos are rejected
        result = self.normalizer.normalize("Pytthon")
        self.assertIsNone(result)

    def test_fuzzy_enabled_matches_minor_typo(self):
        # "Pytthon" -> 1 extra 't', ratio > 0.85 with "python"
        result = self.normalizer.normalize("Pytthon", allow_fuzzy=True)
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "Python")
        self.assertGreaterEqual(result["confidence"], 0.80)
        self.assertLess(result["confidence"], 1.0)
        self.assertEqual(result["method"], "fuzzy")

    def test_fuzzy_does_not_match_completely_different_term(self):
        result = self.normalizer.normalize("Photoshop", allow_fuzzy=True)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test Suite 5: Deduplication in normalize_list & to_canonical_list
# ---------------------------------------------------------------------------

class TestListNormalizationAndDeduplication(unittest.TestCase):
    """Tests list normalization and deduplication per job posting."""

    def setUp(self):
        self.normalizer = _make_normalizer()

    def test_normalize_list_deduplicates_same_canonical_skill(self):
        raw_skills = ["Python", "python3", "py", "React.js", "React", "reactjs"]
        result = self.normalizer.normalize_list(raw_skills, deduplicate=True)
        canonical_names = [r["canonical_name"] for r in result]
        self.assertEqual(len(canonical_names), 2)
        self.assertIn("Python", canonical_names)
        self.assertIn("React", canonical_names)

    def test_normalize_list_filters_unknown_skills(self):
        raw_skills = ["Python", "UnknownSkillA", "Docker", "UnknownSkillB"]
        result = self.normalizer.normalize_list(raw_skills)
        canonical_names = [r["canonical_name"] for r in result]
        self.assertEqual(canonical_names, ["Python", "Docker"])

    def test_normalize_list_without_deduplication(self):
        raw_skills = ["Python", "python3"]
        result = self.normalizer.normalize_list(raw_skills, deduplicate=False)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["canonical_name"], "Python")
        self.assertEqual(result[1]["canonical_name"], "Python")

    def test_normalize_list_empty_input(self):
        self.assertEqual(self.normalizer.normalize_list([]), [])

    def test_to_canonical_list_returns_sorted_strings(self):
        raw_skills = ["React.js", "docker-compose", "python3", "Fast API"]
        canonicals = self.normalizer.to_canonical_list(raw_skills)
        # Expected sorted: Docker, FastAPI, Python, React
        self.assertEqual(canonicals, ["Docker", "FastAPI", "Python", "React"])

    def test_to_canonical_list_duplicate_variants(self):
        raw_skills = ["React.js", "React", "reactjs", "ReactJS"]
        canonicals = self.normalizer.to_canonical_list(raw_skills)
        self.assertEqual(canonicals, ["React"])

    def test_to_canonical_list_empty_input(self):
        self.assertEqual(self.normalizer.to_canonical_list([]), [])


# ---------------------------------------------------------------------------
# Test Suite 6: Batch Normalization
# ---------------------------------------------------------------------------

class TestBatchNormalization(unittest.TestCase):
    """Tests batch normalization methods across multiple job postings."""

    def setUp(self):
        self.normalizer = _make_normalizer()

    def test_normalize_batch_length_and_order(self):
        batch = [
            ["Python", "FastAPI"],
            ["React.js", "react-js"],
            [],
            ["Docker"],
        ]
        result = self.normalizer.normalize_batch(batch)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 1)  # deduplicated React
        self.assertEqual(len(result[2]), 0)  # empty list
        self.assertEqual(len(result[3]), 1)

    def test_batch_to_canonical_lists(self):
        batch = [
            ["Python", "python3", "Fast API"],
            ["React.js", "Dockerfile"],
        ]
        result = self.normalizer.batch_to_canonical_lists(batch)
        self.assertEqual(result[0], ["FastAPI", "Python"])
        self.assertEqual(result[1], ["Docker", "React"])

    def test_batch_empty_input(self):
        self.assertEqual(self.normalizer.normalize_batch([]), [])
        self.assertEqual(self.normalizer.batch_to_canonical_lists([]), [])


# ---------------------------------------------------------------------------
# Test Suite 7: Module-level Convenience Functions & Real Taxonomy
# ---------------------------------------------------------------------------

class TestRealTaxonomyAndModuleLevelAPI(unittest.TestCase):
    """Tests default normalizer with real skill_taxonomy.json from disk."""

    def test_default_normalizer_loads_real_taxonomy(self):
        normalizer = get_default_normalizer()
        self.assertGreater(len(normalizer._canonical_to_entry), 50)

    def test_module_level_normalize_known_skills(self):
        self.assertEqual(normalize("react.js")["canonical_name"], "React")
        self.assertEqual(normalize("python3")["canonical_name"], "Python")
        self.assertEqual(normalize("NodeJS")["canonical_name"], "Node.js")
        self.assertEqual(normalize("PostgreSQL")["canonical_name"], "PostgreSQL")
        self.assertEqual(normalize("postgres")["canonical_name"], "PostgreSQL")

    def test_module_level_to_canonical_list_contract_a_shape(self):
        raw = ["React.js", "fast-api", "docker-compose", "unknown_xyz"]
        canonicals = to_canonical_list(raw)
        self.assertEqual(canonicals, ["Docker", "FastAPI", "React"])
        # Assert each element is pure string
        for c in canonicals:
            self.assertIsInstance(c, str)


if __name__ == "__main__":
    unittest.main()
