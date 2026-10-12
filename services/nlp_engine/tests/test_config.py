"""
test_config.py — Phase 0 Test Checkpoint
==========================================
Verifies that config.py is correctly structured and all paths resolve.
Run with: python -m unittest services/nlp_engine/tests/test_config.py
"""

import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path so imports work
# regardless of where the test is run from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # e:\Skill_Tracker_
sys.path.insert(0, str(PROJECT_ROOT))


class TestConfig(unittest.TestCase):
    """Phase 0 — Validate config.py imports cleanly and values are correct."""

    def setUp(self):
        """Import config fresh for each test."""
        import importlib
        import services.nlp_engine.config as cfg
        importlib.reload(cfg)  # ensures no cached state between tests
        self.cfg = cfg

    # --- Path Tests ---

    def test_nlp_engine_dir_exists(self):
        """NLP_ENGINE_DIR must point to an existing directory."""
        self.assertTrue(
            self.cfg.NLP_ENGINE_DIR.exists(),
            f"NLP_ENGINE_DIR does not exist: {self.cfg.NLP_ENGINE_DIR}"
        )

    def test_taxonomy_dir_exists(self):
        """TAXONOMY_DIR must point to an existing directory."""
        self.assertTrue(
            self.cfg.TAXONOMY_DIR.exists(),
            f"TAXONOMY_DIR does not exist: {self.cfg.TAXONOMY_DIR}"
        )

    def test_tests_dir_exists(self):
        """TESTS_DIR must point to an existing directory."""
        self.assertTrue(
            self.cfg.TESTS_DIR.exists(),
            f"TESTS_DIR does not exist: {self.cfg.TESTS_DIR}"
        )

    def test_taxonomy_file_path_is_correct_type(self):
        """TAXONOMY_FILE must be a Path object pointing to a .json file."""
        self.assertIsInstance(self.cfg.TAXONOMY_FILE, Path)
        self.assertEqual(
            self.cfg.TAXONOMY_FILE.suffix, ".json",
            "TAXONOMY_FILE must have a .json extension"
        )

    # --- Type & Value Tests ---

    def test_spacy_model_is_string(self):
        """SPACY_MODEL must be a non-empty string."""
        self.assertIsInstance(self.cfg.SPACY_MODEL, str)
        self.assertGreater(len(self.cfg.SPACY_MODEL), 0)

    def test_spacy_model_valid_name(self):
        """SPACY_MODEL should be a known valid spaCy model name."""
        valid_models = {"en_core_web_sm", "en_core_web_md", "en_core_web_lg"}
        self.assertIn(
            self.cfg.SPACY_MODEL, valid_models,
            f"Unexpected SPACY_MODEL value: '{self.cfg.SPACY_MODEL}'"
        )

    def test_skill_entity_label_is_string(self):
        """SKILL_ENTITY_LABEL must be a non-empty uppercase string."""
        self.assertIsInstance(self.cfg.SKILL_ENTITY_LABEL, str)
        self.assertEqual(
            self.cfg.SKILL_ENTITY_LABEL,
            self.cfg.SKILL_ENTITY_LABEL.upper(),
            "SKILL_ENTITY_LABEL should be uppercase (spaCy convention)"
        )

    def test_rule_based_confidence_is_float(self):
        """RULE_BASED_CONFIDENCE must be a float between 0 and 1."""
        self.assertIsInstance(self.cfg.RULE_BASED_CONFIDENCE, float)
        self.assertGreaterEqual(self.cfg.RULE_BASED_CONFIDENCE, 0.0)
        self.assertLessEqual(self.cfg.RULE_BASED_CONFIDENCE, 1.0)

    def test_min_confidence_threshold_is_float(self):
        """MIN_CONFIDENCE_THRESHOLD must be a float between 0 and 1."""
        self.assertIsInstance(self.cfg.MIN_CONFIDENCE_THRESHOLD, float)
        self.assertGreaterEqual(self.cfg.MIN_CONFIDENCE_THRESHOLD, 0.0)
        self.assertLessEqual(self.cfg.MIN_CONFIDENCE_THRESHOLD, 1.0)

    def test_confidence_ordering(self):
        """RULE_BASED_CONFIDENCE must be >= MIN_CONFIDENCE_THRESHOLD."""
        self.assertGreaterEqual(
            self.cfg.RULE_BASED_CONFIDENCE,
            self.cfg.MIN_CONFIDENCE_THRESHOLD,
            "RULE_BASED_CONFIDENCE should be >= MIN_CONFIDENCE_THRESHOLD"
        )

    def test_min_cooccurrence_count_is_positive_int(self):
        """MIN_COOCCURRENCE_COUNT must be a positive integer."""
        self.assertIsInstance(self.cfg.MIN_COOCCURRENCE_COUNT, int)
        self.assertGreater(self.cfg.MIN_COOCCURRENCE_COUNT, 0)

    def test_date_format_is_string(self):
        """DATE_FORMAT must be a non-empty string."""
        self.assertIsInstance(self.cfg.DATE_FORMAT, str)
        self.assertGreater(len(self.cfg.DATE_FORMAT), 0)

    def test_log_level_is_valid(self):
        """LOG_LEVEL must be one of the standard Python logging levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        self.assertIn(
            self.cfg.LOG_LEVEL, valid_levels,
            f"Invalid LOG_LEVEL: '{self.cfg.LOG_LEVEL}'"
        )

    def test_taxonomy_required_keys(self):
        """TAXONOMY_REQUIRED_KEYS must be a tuple containing expected keys."""
        self.assertIsInstance(self.cfg.TAXONOMY_REQUIRED_KEYS, tuple)
        for key in ("canonical_name", "category", "aliases"):
            self.assertIn(
                key, self.cfg.TAXONOMY_REQUIRED_KEYS,
                f"Missing required taxonomy key: '{key}'"
            )

    def test_spacy_disabled_components_is_list(self):
        """SPACY_DISABLED_COMPONENTS must be a list."""
        self.assertIsInstance(self.cfg.SPACY_DISABLED_COMPONENTS, list)

    # --- Validate Config Function Test ---

    def test_validate_config_does_not_raise(self):
        """_validate_config() must not raise for a correctly set-up project."""
        try:
            self.cfg._validate_config()
        except Exception as e:
            self.fail(f"_validate_config() raised unexpectedly: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
