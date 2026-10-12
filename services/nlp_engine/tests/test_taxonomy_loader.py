"""
test_taxonomy_loader.py — Phase 1 Test Checkpoint
===================================================
Validates taxonomy_loader.py in full isolation using:
  - The real skill_taxonomy.json for integration-style tests.
  - Temp in-memory JSON for unit/edge-case tests (no filesystem side effects).

Run with: python -m unittest services/nlp_engine/tests/test_taxonomy_loader.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — project root on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # e:\Skill_Tracker_
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp_engine.taxonomy.taxonomy_loader import (
    load_taxonomy,
    reload_taxonomy,
    get_alias_to_canonical,
    get_canonical_to_entry,
    get_all_canonical_names,
    get_categories,
    resolve_alias,
)


# ---------------------------------------------------------------------------
# Helper: write a temp JSON taxonomy file
# ---------------------------------------------------------------------------

def _write_temp_taxonomy(entries: list, suffix: str = ".json") -> Path:
    """Writes a list of entries to a temp file and returns its Path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    )
    json.dump(entries, tmp)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestTaxonomyLoaderWithRealFile(unittest.TestCase):
    """Integration tests — uses the actual skill_taxonomy.json."""

    def setUp(self):
        """Force reload before each test to ensure clean state."""
        reload_taxonomy()

    def test_load_returns_list(self):
        """load_taxonomy() must return a list."""
        result = load_taxonomy()
        self.assertIsInstance(result, list)

    def test_loaded_list_is_not_empty(self):
        """Taxonomy must contain at least one entry."""
        result = load_taxonomy()
        self.assertGreater(len(result), 0)

    def test_every_entry_has_required_keys(self):
        """Every entry in the loaded taxonomy must have all required keys."""
        required = {"id", "canonical_name", "category", "aliases"}
        for entry in load_taxonomy():
            for key in required:
                self.assertIn(
                    key, entry,
                    f"Entry '{entry.get('canonical_name', '?')}' missing key '{key}'"
                )

    def test_aliases_are_lists(self):
        """Every entry's 'aliases' must be a list."""
        for entry in load_taxonomy():
            self.assertIsInstance(
                entry["aliases"], list,
                f"'aliases' for '{entry['canonical_name']}' must be a list"
            )

    def test_canonical_names_are_unique(self):
        """No two entries should share the same canonical_name."""
        names = [e["canonical_name"] for e in load_taxonomy()]
        self.assertEqual(len(names), len(set(names)), "Duplicate canonical_names found!")

    def test_idempotent_load(self):
        """Calling load_taxonomy() twice returns the same list object (cache hit)."""
        first = load_taxonomy()
        second = load_taxonomy()
        self.assertIs(first, second, "load_taxonomy() should return cached object on 2nd call")

    # --- Alias lookup tests ---

    def test_resolve_exact_canonical(self):
        """Canonical name itself should resolve to itself."""
        self.assertEqual(resolve_alias("Python"), "Python")
        self.assertEqual(resolve_alias("React"), "React")
        self.assertEqual(resolve_alias("Docker"), "Docker")

    def test_resolve_alias_case_insensitive(self):
        """Alias resolution must be case-insensitive."""
        self.assertEqual(resolve_alias("python3"), "Python")
        self.assertEqual(resolve_alias("PYTHON3"), "Python")
        self.assertEqual(resolve_alias("Python3"), "Python")

    def test_resolve_known_variant_react(self):
        """'React.js', 'ReactJS', 'reactjs' must all resolve to 'React'."""
        for variant in ["React.js", "ReactJS", "react-js", "reactjs", "React JS"]:
            with self.subTest(variant=variant):
                self.assertEqual(resolve_alias(variant), "React")

    def test_resolve_known_variant_nodejs(self):
        """'NodeJS', 'Node JS', 'nodejs' must all resolve to 'Node.js'."""
        for variant in ["NodeJS", "Node JS", "node-js", "nodejs"]:
            with self.subTest(variant=variant):
                self.assertEqual(resolve_alias(variant), "Node.js")

    def test_resolve_known_variant_fastapi(self):
        """'fast-api', 'Fast API' must resolve to 'FastAPI'."""
        for variant in ["fast-api", "Fast API", "fastapi-python"]:
            with self.subTest(variant=variant):
                self.assertEqual(resolve_alias(variant), "FastAPI")

    def test_resolve_unknown_returns_none(self):
        """Unknown skill strings must return None (not raise)."""
        self.assertIsNone(resolve_alias("UnknownTool_XYZ_2099"))
        self.assertIsNone(resolve_alias(""))
        self.assertIsNone(resolve_alias("   "))

    # --- Lookup table structure ---

    def test_alias_to_canonical_is_dict(self):
        """get_alias_to_canonical() must return a non-empty dict."""
        result = get_alias_to_canonical()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_alias_to_canonical_keys_are_lowercase(self):
        """All keys in alias_to_canonical must be lowercase strings."""
        for key in get_alias_to_canonical().keys():
            self.assertEqual(key, key.lower(), f"Key '{key}' is not lowercase")

    def test_canonical_to_entry_is_dict(self):
        """get_canonical_to_entry() must return a non-empty dict."""
        result = get_canonical_to_entry()
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_get_all_canonical_names_sorted(self):
        """get_all_canonical_names() must return a sorted list of strings."""
        names = get_all_canonical_names()
        self.assertIsInstance(names, list)
        self.assertEqual(names, sorted(names), "canonical names must be sorted")

    def test_get_categories_returns_list_of_strings(self):
        """get_categories() must return a sorted list of non-empty strings."""
        categories = get_categories()
        self.assertIsInstance(categories, list)
        self.assertGreater(len(categories), 0)
        for cat in categories:
            self.assertIsInstance(cat, str)
            self.assertGreater(len(cat.strip()), 0)

    def test_expected_categories_present(self):
        """Known categories from context.md taxonomy must be present."""
        expected = {"Language", "Frontend", "Backend", "Database", "DevOps", "Cloud"}
        actual = set(get_categories())
        for cat in expected:
            self.assertIn(cat, actual, f"Expected category '{cat}' not found")

    # --- Cross-team compatibility ---

    def test_contract_critical_skills_exist(self):
        """
        Skills explicitly mentioned in context.md contracts and DB schema
        must exist as canonical names.
        """
        critical_skills = ["Python", "FastAPI", "Docker"]
        canonical_names = set(get_all_canonical_names())
        for skill in critical_skills:
            self.assertIn(
                skill, canonical_names,
                f"Critical skill '{skill}' from Contract A not found in taxonomy!"
            )

    def test_db_schema_skills_exist(self):
        """Skills mentioned in context.md DB schema / gap report must exist."""
        db_skills = ["Docker", "Kubernetes", "gRPC", "FastAPI", "React"]
        canonical_names = set(get_all_canonical_names())
        for skill in db_skills:
            self.assertIn(
                skill, canonical_names,
                f"DB/report skill '{skill}' not found in taxonomy!"
            )


class TestTaxonomyLoaderEdgeCases(unittest.TestCase):
    """Unit tests for edge cases using temporary JSON files."""

    def _reload_with(self, entries: list):
        """Helper: write entries to temp file and reload taxonomy from it."""
        tmp = _write_temp_taxonomy(entries)
        try:
            result = reload_taxonomy(tmp)
        finally:
            tmp.unlink(missing_ok=True)
        return result

    def test_file_not_found_raises(self):
        """Loading from a non-existent path must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            reload_taxonomy(Path("/nonexistent/path/taxonomy.json"))

    def test_empty_list_raises_value_error(self):
        """A taxonomy JSON file with an empty array must raise ValueError."""
        tmp = _write_temp_taxonomy([])
        try:
            with self.assertRaises(ValueError):
                reload_taxonomy(tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def test_missing_required_key_entry_is_skipped(self):
        """
        An entry missing 'aliases' should be skipped (logged),
        while valid entries still load correctly.
        """
        entries = [
            {"id": "x-1", "canonical_name": "ValidSkill", "category": "Test", "aliases": ["valid"]},
            {"id": "x-2", "canonical_name": "BadSkill", "category": "Test"},   # missing 'aliases'
        ]
        result = self._reload_with(entries)
        names = [e["canonical_name"] for e in result]
        self.assertIn("ValidSkill", names)
        self.assertNotIn("BadSkill", names)

    def test_empty_canonical_name_is_skipped(self):
        """An entry with an empty canonical_name must be skipped."""
        entries = [
            {"id": "x-1", "canonical_name": "", "category": "Test", "aliases": []},
            {"id": "x-2", "canonical_name": "RealSkill", "category": "Test", "aliases": ["rs"]},
        ]
        result = self._reload_with(entries)
        names = [e["canonical_name"] for e in result]
        self.assertIn("RealSkill", names)
        self.assertNotIn("", names)

    def test_alias_conflict_does_not_crash(self):
        """
        When two entries claim the same alias, the first mapping wins
        and no exception is raised.
        """
        entries = [
            {"id": "x-1", "canonical_name": "SkillA", "category": "Test", "aliases": ["shared-alias"]},
            {"id": "x-2", "canonical_name": "SkillB", "category": "Test", "aliases": ["shared-alias"]},
        ]
        try:
            result = self._reload_with(entries)
        except Exception as e:
            self.fail(f"Alias conflict caused an unexpected exception: {e}")
        # First mapping (SkillA) must win
        lookup = get_alias_to_canonical()
        self.assertEqual(lookup.get("shared-alias"), "SkillA")

    def test_whitespace_in_alias_is_stripped(self):
        """Aliases with surrounding whitespace should still resolve correctly."""
        entries = [
            {"id": "x-1", "canonical_name": "CleanSkill", "category": "Test", "aliases": ["  trimmed  "]},
        ]
        self._reload_with(entries)
        self.assertEqual(resolve_alias("trimmed"), "CleanSkill")

    def test_reload_clears_old_data(self):
        """reload_taxonomy() must replace old data entirely."""
        # Load a small custom taxonomy
        small = [
            {"id": "x-1", "canonical_name": "TempSkill", "category": "Test", "aliases": ["ts"]}
        ]
        self._reload_with(small)
        names_after_small = get_all_canonical_names()
        self.assertIn("TempSkill", names_after_small)

        # Reload the real taxonomy — TempSkill should be gone
        reload_taxonomy()
        names_after_real = get_all_canonical_names()
        self.assertNotIn("TempSkill", names_after_real)


if __name__ == "__main__":
    unittest.main(verbosity=2)
