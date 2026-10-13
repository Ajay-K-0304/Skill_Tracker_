"""
test_ner_pipeline.py — Phase 3 Test Checkpoint
================================================
Tests entity_ruler.py and ner_pipeline.py in full isolation.

KEY DESIGN: All tests use spacy.blank("en") — no downloaded model required.
  spacy.blank("en") provides only a tokenizer, which is all we need to
  verify EntityRuler pattern matching without any network/disk dependency.

Run with: python -m unittest services/nlp_engine/tests/test_ner_pipeline.py -v
"""

import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from services.nlp_engine.pipeline.entity_ruler import build_patterns, attach_entity_ruler
from services.nlp_engine.pipeline.ner_pipeline import NERPipeline
from services.nlp_engine.taxonomy.taxonomy_loader import reload_taxonomy


# ---------------------------------------------------------------------------
# Shared fixtures
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
]


def _make_pipeline(taxonomy=None) -> NERPipeline:
    """Creates a NERPipeline with blank spaCy model and mock taxonomy."""
    nlp = spacy.blank("en")
    entries = taxonomy if taxonomy is not None else MOCK_TAXONOMY
    return NERPipeline(nlp=nlp, taxonomy_entries=entries)


# ---------------------------------------------------------------------------
# Tests: build_patterns()
# ---------------------------------------------------------------------------

@unittest.skipUnless(SPACY_AVAILABLE, "spaCy not installed")
class TestBuildPatterns(unittest.TestCase):
    """Tests for entity_ruler.build_patterns()."""

    def test_returns_list(self):
        patterns = build_patterns(MOCK_TAXONOMY)
        self.assertIsInstance(patterns, list)

    def test_pattern_count_gte_taxonomy_size(self):
        """Each entry produces at least 1 pattern (more with aliases)."""
        patterns = build_patterns(MOCK_TAXONOMY)
        self.assertGreaterEqual(len(patterns), len(MOCK_TAXONOMY))

    def test_pattern_has_required_keys(self):
        """Every pattern must have label, pattern, id keys."""
        for p in build_patterns(MOCK_TAXONOMY):
            self.assertIn("label", p)
            self.assertIn("pattern", p)
            self.assertIn("id", p)

    def test_all_labels_are_skill(self):
        """Every pattern label must be 'SKILL'."""
        for p in build_patterns(MOCK_TAXONOMY):
            self.assertEqual(p["label"], "SKILL")

    def test_canonical_names_included(self):
        """Canonical name of each entry must appear as a pattern."""
        patterns = build_patterns(MOCK_TAXONOMY)
        pattern_strings = {p["pattern"] for p in patterns}
        for entry in MOCK_TAXONOMY:
            self.assertIn(
                entry["canonical_name"], pattern_strings,
                f"Canonical '{entry['canonical_name']}' missing from patterns"
            )

    def test_aliases_included(self):
        """Each alias must appear as a separate pattern."""
        patterns = build_patterns(MOCK_TAXONOMY)
        pattern_strings = {p["pattern"] for p in patterns}
        for entry in MOCK_TAXONOMY:
            for alias in entry["aliases"]:
                self.assertIn(
                    alias, pattern_strings,
                    f"Alias '{alias}' missing from patterns"
                )

    def test_taxonomy_ids_in_patterns(self):
        """Pattern id must match the taxonomy entry id."""
        patterns = build_patterns(MOCK_TAXONOMY)
        tax_ids = {e["id"] for e in MOCK_TAXONOMY}
        for p in patterns:
            self.assertIn(
                p["id"], tax_ids,
                f"Unknown taxonomy id '{p['id']}' found in pattern"
            )

    def test_no_duplicate_patterns(self):
        """No two patterns should have the same (pattern_lower, id) pair."""
        patterns = build_patterns(MOCK_TAXONOMY)
        seen = set()
        for p in patterns:
            key = (p["pattern"].lower(), p["id"])
            self.assertNotIn(key, seen, f"Duplicate pattern: {p}")
            seen.add(key)

    def test_empty_aliases_handled(self):
        """Entry with empty aliases list should still produce 1 pattern (canonical)."""
        entries = [{"id": "x-1", "canonical_name": "Go", "category": "Language", "aliases": []}]
        patterns = build_patterns(entries)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0]["pattern"], "Go")

    def test_blank_alias_strings_skipped(self):
        """Aliases that are blank strings should not generate patterns."""
        entries = [
            {"id": "x-1", "canonical_name": "Go", "category": "Language", "aliases": ["", "  ", "Golang"]}
        ]
        patterns = build_patterns(entries)
        pattern_strings = [p["pattern"] for p in patterns]
        self.assertNotIn("", pattern_strings)
        self.assertNotIn("  ", pattern_strings)
        self.assertIn("Golang", pattern_strings)


# ---------------------------------------------------------------------------
# Tests: attach_entity_ruler()
# ---------------------------------------------------------------------------

@unittest.skipUnless(SPACY_AVAILABLE, "spaCy not installed")
class TestAttachEntityRuler(unittest.TestCase):
    """Tests for entity_ruler.attach_entity_ruler()."""

    def test_entity_ruler_in_pipe_names(self):
        """After attachment, 'entity_ruler' must appear in nlp.pipe_names."""
        nlp = spacy.blank("en")
        attach_entity_ruler(nlp, taxonomy_entries=MOCK_TAXONOMY)
        self.assertIn("entity_ruler", nlp.pipe_names)

    def test_entity_ruler_is_first(self):
        """EntityRuler must be the first component in the pipeline."""
        nlp = spacy.blank("en")
        attach_entity_ruler(nlp, taxonomy_entries=MOCK_TAXONOMY)
        self.assertEqual(nlp.pipe_names[0], "entity_ruler")

    def test_empty_taxonomy_raises(self):
        """Attaching with an empty taxonomy must raise ValueError."""
        nlp = spacy.blank("en")
        with self.assertRaises(ValueError):
            attach_entity_ruler(nlp, taxonomy_entries=[])


# ---------------------------------------------------------------------------
# Tests: NERPipeline — construction
# ---------------------------------------------------------------------------

@unittest.skipUnless(SPACY_AVAILABLE, "spaCy not installed")
class TestNERPipelineConstruction(unittest.TestCase):

    def test_pipeline_creates_successfully(self):
        """NERPipeline must initialise without errors using blank model."""
        try:
            pipeline = _make_pipeline()
        except Exception as e:
            self.fail(f"NERPipeline() raised unexpectedly: {e}")

    def test_pipe_names_contains_entity_ruler(self):
        pipeline = _make_pipeline()
        self.assertIn("entity_ruler", pipeline.pipe_names)

    def test_pipe_names_property_returns_list(self):
        pipeline = _make_pipeline()
        self.assertIsInstance(pipeline.pipe_names, list)


# ---------------------------------------------------------------------------
# Tests: NERPipeline.extract() — skill detection
# ---------------------------------------------------------------------------

@unittest.skipUnless(SPACY_AVAILABLE, "spaCy not installed")
class TestNERPipelineExtract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = _make_pipeline()

    # --- Return type & structure ---

    def test_extract_returns_list(self):
        result = self.pipeline.extract("We need Python developers.")
        self.assertIsInstance(result, list)

    def test_each_hit_has_required_keys(self):
        """Every hit dict must contain all expected keys."""
        result = self.pipeline.extract("We need Python and Docker.")
        required_keys = {"text", "label", "start_char", "end_char", "taxonomy_id"}
        for hit in result:
            for key in required_keys:
                self.assertIn(key, hit, f"Hit missing key '{key}': {hit}")

    def test_hit_label_is_skill(self):
        result = self.pipeline.extract("Python and FastAPI are required.")
        for hit in result:
            self.assertEqual(hit["label"], "SKILL")

    def test_char_offsets_are_ints(self):
        result = self.pipeline.extract("Python is great.")
        for hit in result:
            self.assertIsInstance(hit["start_char"], int)
            self.assertIsInstance(hit["end_char"], int)

    def test_char_offsets_are_valid(self):
        """start_char < end_char and both within text bounds."""
        text = "We need Python and Docker."
        result = self.pipeline.extract(text)
        for hit in result:
            self.assertLess(hit["start_char"], hit["end_char"])
            self.assertGreaterEqual(hit["start_char"], 0)
            self.assertLessEqual(hit["end_char"], len(text))

    # --- Exact canonical name detection ---

    def test_detects_canonical_python(self):
        hits = self.pipeline.extract("We need Python developers.")
        texts = [h["text"].lower() for h in hits]
        self.assertIn("python", texts)

    def test_detects_canonical_docker(self):
        hits = self.pipeline.extract("Docker experience required.")
        texts = [h["text"].lower() for h in hits]
        self.assertIn("docker", texts)

    def test_detects_canonical_fastapi(self):
        hits = self.pipeline.extract("Build APIs using FastAPI.")
        texts = [h["text"].lower() for h in hits]
        self.assertIn("fastapi", texts)

    # --- Alias detection (case variants) ---

    def test_detects_alias_reactjs(self):
        """'ReactJS' alias must match and resolve to React taxonomy id."""
        hits = self.pipeline.extract("3 years of ReactJS experience.")
        ids = [h["taxonomy_id"] for h in hits]
        self.assertIn("t-101", ids)

    def test_detects_alias_react_dot_js(self):
        """'React.js' alias must be detected."""
        hits = self.pipeline.extract("Experience with React.js preferred.")
        ids = [h["taxonomy_id"] for h in hits]
        self.assertIn("t-101", ids)

    def test_detects_alias_python3(self):
        """'python3' alias must match Python."""
        hits = self.pipeline.extract("Strong python3 skills required.")
        ids = [h["taxonomy_id"] for h in hits]
        self.assertIn("t-001", ids)

    def test_case_insensitive_match(self):
        """PYTHON (uppercase) must match the Python taxonomy entry."""
        hits = self.pipeline.extract("Experience in PYTHON required.")
        ids = [h["taxonomy_id"] for h in hits]
        self.assertIn("t-001", ids)

    # --- Multi-skill detection ---

    def test_detects_multiple_skills(self):
        """Multiple skills in one sentence must all be detected."""
        text = "We need Python, FastAPI, and Docker for this role."
        hits = self.pipeline.extract(text)
        ids = {h["taxonomy_id"] for h in hits}
        self.assertIn("t-001", ids)  # Python
        self.assertIn("t-201", ids)  # FastAPI
        self.assertIn("t-401", ids)  # Docker

    def test_detects_skills_in_full_jd(self):
        """Real-world style job description with multiple skills."""
        text = (
            "We are looking for a backend engineer with 3+ years of Python "
            "experience. You should know FastAPI or Docker. React.js is a plus."
        )
        hits = self.pipeline.extract(text)
        ids = {h["taxonomy_id"] for h in hits}
        self.assertIn("t-001", ids)   # Python
        self.assertIn("t-201", ids)   # FastAPI
        self.assertIn("t-401", ids)   # Docker
        self.assertIn("t-101", ids)   # React (via React.js alias)

    # --- Deduplication ---

    def test_same_skill_mentioned_twice_returned_once(self):
        """A skill mentioned multiple times in one text should appear once."""
        hits = self.pipeline.extract("Python Python Python Python.")
        python_hits = [h for h in hits if h["taxonomy_id"] == "t-001"]
        self.assertEqual(len(python_hits), 1)

    def test_alias_and_canonical_same_skill_deduplicated(self):
        """If both 'Python' and 'python3' appear, only one hit per taxonomy_id."""
        hits = self.pipeline.extract("Python and python3 are both mentioned here.")
        python_hits = [h for h in hits if h["taxonomy_id"] == "t-001"]
        self.assertEqual(len(python_hits), 1)

    # --- Unknown words ---

    def test_unknown_skill_not_detected(self):
        """Words not in the taxonomy must not appear in results."""
        hits = self.pipeline.extract("We use UnknownTool2099 and MysteriousLib.")
        self.assertEqual(hits, [])

    def test_partial_word_not_matched(self):
        """'Pyth' should not match 'Python'."""
        hits = self.pipeline.extract("Pyth on is not Python.")
        # "Python" should still be found, "Pyth" alone should not
        texts = [h["text"].lower() for h in hits]
        self.assertNotIn("pyth", texts)

    # --- Edge cases ---

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.pipeline.extract(""), [])

    def test_whitespace_only_returns_empty(self):
        self.assertEqual(self.pipeline.extract("   "), [])

    def test_no_skills_in_text_returns_empty(self):
        hits = self.pipeline.extract("The quick brown fox jumps over the lazy dog.")
        self.assertEqual(hits, [])

    def test_taxonomy_id_in_hit_matches_taxonomy(self):
        """The taxonomy_id in each hit must correspond to a real taxonomy entry."""
        valid_ids = {e["id"] for e in MOCK_TAXONOMY}
        hits = self.pipeline.extract("Python FastAPI Docker React.")
        for hit in hits:
            self.assertIn(
                hit["taxonomy_id"], valid_ids,
                f"Hit has unknown taxonomy_id: {hit}"
            )


# ---------------------------------------------------------------------------
# Tests: NERPipeline.extract_batch()
# ---------------------------------------------------------------------------

@unittest.skipUnless(SPACY_AVAILABLE, "spaCy not installed")
class TestNERPipelineExtractBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = _make_pipeline()

    def test_batch_returns_list(self):
        result = self.pipeline.extract_batch(["Python dev", "FastAPI engineer"])
        self.assertIsInstance(result, list)

    def test_batch_length_matches_input(self):
        """Output list must be same length as input list."""
        texts = ["Python dev", "FastAPI needed", "React.js expert"]
        result = self.pipeline.extract_batch(texts)
        self.assertEqual(len(result), len(texts))

    def test_batch_preserves_order(self):
        """Results must align with their corresponding input text."""
        texts = ["Python developer", "Docker expert"]
        results = self.pipeline.extract_batch(texts)
        # First result should contain Python
        ids_0 = {h["taxonomy_id"] for h in results[0]}
        self.assertIn("t-001", ids_0)
        # Second result should contain Docker
        ids_1 = {h["taxonomy_id"] for h in results[1]}
        self.assertIn("t-401", ids_1)

    def test_batch_empty_list_returns_empty(self):
        self.assertEqual(self.pipeline.extract_batch([]), [])

    def test_batch_handles_empty_strings(self):
        """Empty strings in batch should produce empty hit lists."""
        results = self.pipeline.extract_batch(["Python dev", "", "FastAPI"])
        self.assertEqual(results[1], [])

    def test_batch_results_are_lists_of_dicts(self):
        """Each element of the batch result must be a list of dicts."""
        results = self.pipeline.extract_batch(["Python", "Docker"])
        for res in results:
            self.assertIsInstance(res, list)
            for hit in res:
                self.assertIsInstance(hit, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
