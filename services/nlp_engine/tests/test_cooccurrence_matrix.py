"""
test_cooccurrence_matrix.py — Phase 5 Test Checkpoint
======================================================
Tests cooccurrence_matrix.py in full isolation.

Coverage:
  1. Matrix symmetry: matrix[A, B] == matrix[B, A] for all (A, B).
  2. Diagonal is zero: matrix[A, A] == 0 for all A.
  3. Correct frequency count on deterministic fixture data.
  4. Deduplication of skills within a single job posting.
  5. Handling empty input, single-skill jobs, None, and whitespace entries.
  6. Minimum count filtering (min_count threshold).
  7. Predefined fixed vocabulary support.
  8. Query helpers: get_count() and get_top_cooccurring().
  9. Incremental update() accumulating counts correctly.
  10. Serializers: to_dict(), to_edges(), to_csv(), to_json().
  11. Module-level convenience functions.

Run with: python -m unittest services/nlp_engine/tests/test_cooccurrence_matrix.py -v
"""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.nlp_engine.cooccurrence.cooccurrence_matrix import (
    CooccurrenceMatrix,
    build_cooccurrence_matrix,
    build_network_edges,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

SAMPLE_POSTINGS = [
    ["Python", "FastAPI", "Docker"],
    ["Python", "FastAPI"],
    ["Python", "React"],
    ["Docker", "React"],
]


# ---------------------------------------------------------------------------
# Test Suite 1: Matrix Shape, Symmetry, and Diagonal
# ---------------------------------------------------------------------------

class TestMatrixStructure(unittest.TestCase):
    """Tests fundamental mathematical properties of the co-occurrence matrix."""

    def setUp(self):
        self.model = CooccurrenceMatrix()
        self.df = self.model.fit(SAMPLE_POSTINGS).matrix

    def test_matrix_is_symmetric(self):
        """matrix.loc[A, B] must equal matrix.loc[B, A] for all pairs."""
        for skill_a in self.df.index:
            for skill_b in self.df.columns:
                self.assertEqual(
                    self.df.loc[skill_a, skill_b],
                    self.df.loc[skill_b, skill_a],
                    f"Asymmetry detected between {skill_a} and {skill_b}",
                )

    def test_diagonal_is_strictly_zero(self):
        """No skill can co-occur with itself; diagonal must be 0."""
        for skill in self.df.index:
            self.assertEqual(
                self.df.loc[skill, skill],
                0,
                f"Non-zero diagonal for skill '{skill}'",
            )

    def test_row_and_column_labels_match(self):
        """Index and columns must be identical and sorted."""
        self.assertListEqual(list(self.df.index), list(self.df.columns))
        self.assertListEqual(list(self.df.index), sorted(list(self.df.index)))

    def test_vocabulary_contains_all_skills(self):
        """All unique skills across the postings must be present."""
        expected = sorted(["Docker", "FastAPI", "Python", "React"])
        self.assertEqual(list(self.df.index), expected)


# ---------------------------------------------------------------------------
# Test Suite 2: Frequency Count Accuracy
# ---------------------------------------------------------------------------

class TestCountAccuracy(unittest.TestCase):
    """Verifies pairwise frequencies match expected occurrences."""

    def setUp(self):
        self.model = CooccurrenceMatrix().fit(SAMPLE_POSTINGS)
        self.df = self.model.matrix

    def test_python_fastapi_pair_count(self):
        # Python & FastAPI appear together in posting 1 and posting 2 -> 2
        self.assertEqual(self.df.loc["Python", "FastAPI"], 2)
        self.assertEqual(self.df.loc["FastAPI", "Python"], 2)

    def test_python_docker_pair_count(self):
        # Python & Docker appear together in posting 1 -> 1
        self.assertEqual(self.df.loc["Python", "Docker"], 1)

    def test_fastapi_react_pair_count(self):
        # FastAPI & React never appear in the same posting -> 0
        self.assertEqual(self.df.loc["FastAPI", "React"], 0)

    def test_docker_react_pair_count(self):
        # Docker & React appear in posting 4 -> 1
        self.assertEqual(self.df.loc["Docker", "React"], 1)

    def test_deduplication_within_single_posting(self):
        """A skill repeated in the same job posting must not inflate pair count."""
        postings = [
            ["Python", "Python", "FastAPI", "FastAPI"],
        ]
        df = CooccurrenceMatrix().fit(postings).matrix
        self.assertEqual(df.loc["Python", "FastAPI"], 1)
        self.assertEqual(df.loc["Python", "Python"], 0)


# ---------------------------------------------------------------------------
# Test Suite 3: Edge Cases & Robustness
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Tests empty inputs, None values, whitespace, and single-skill lists."""

    def test_empty_postings_returns_empty_dataframe(self):
        df = CooccurrenceMatrix().fit([]).matrix
        self.assertEqual(df.shape, (0, 0))

    def test_single_skills_produce_zero_matrix(self):
        """Postings with only 1 skill cannot form pairs -> all 0s."""
        postings = [["Python"], ["Docker"], ["FastAPI"]]
        df = CooccurrenceMatrix().fit(postings).matrix
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(df.values.sum(), 0)

    def test_none_and_empty_strings_ignored(self):
        postings = [
            ["Python", None, "", "   ", "Docker"],
        ]
        df = CooccurrenceMatrix().fit(postings).matrix
        self.assertListEqual(list(df.index), ["Docker", "Python"])
        self.assertEqual(df.loc["Python", "Docker"], 1)

    def test_postings_with_empty_sublists(self):
        postings = [[], ["Python", "Docker"], []]
        df = CooccurrenceMatrix().fit(postings).matrix
        self.assertEqual(df.loc["Python", "Docker"], 1)


# ---------------------------------------------------------------------------
# Test Suite 4: Min Count & Fixed Vocabulary
# ---------------------------------------------------------------------------

class TestThresholdAndVocabulary(unittest.TestCase):
    """Tests min_count filtering and fixed vocabulary dimensions."""

    def test_min_count_filters_rare_pairs(self):
        # In SAMPLE_POSTINGS:
        # Python-FastAPI: 2
        # Others: 1
        model = CooccurrenceMatrix(min_count=2).fit(SAMPLE_POSTINGS)
        df = model.matrix
        self.assertEqual(df.loc["Python", "FastAPI"], 2)
        # All pairs with count 1 are filtered to 0
        self.assertEqual(df.loc["Python", "Docker"], 0)
        self.assertEqual(df.loc["Docker", "React"], 0)

    def test_fixed_vocabulary_preserves_dimensions(self):
        vocab = ["AWS", "Docker", "FastAPI", "Python", "Rust"]
        model = CooccurrenceMatrix(vocabulary=vocab).fit(SAMPLE_POSTINGS)
        df = model.matrix
        self.assertEqual(list(df.index), vocab)
        # Rust and AWS were not in postings -> counts are 0
        self.assertEqual(df.loc["AWS", "Python"], 0)
        self.assertEqual(df.loc["Rust", "FastAPI"], 0)
        # Observed pairs retain their counts
        self.assertEqual(df.loc["Python", "FastAPI"], 2)


# ---------------------------------------------------------------------------
# Test Suite 5: Query Helpers & Incremental Updates
# ---------------------------------------------------------------------------

class TestQueriesAndUpdates(unittest.TestCase):
    """Tests get_count, get_top_cooccurring, and update() workflow."""

    def setUp(self):
        self.model = CooccurrenceMatrix().fit(SAMPLE_POSTINGS)

    def test_get_count_valid(self):
        self.assertEqual(self.model.get_count("Python", "FastAPI"), 2)
        self.assertEqual(self.model.get_count("FastAPI", "Python"), 2)

    def test_get_count_same_skill_returns_zero(self):
        self.assertEqual(self.model.get_count("Python", "Python"), 0)

    def test_get_count_unknown_skill_returns_zero(self):
        self.assertEqual(self.model.get_count("Python", "UnknownSkill"), 0)

    def test_get_top_cooccurring(self):
        top = self.model.get_top_cooccurring("Python", top_n=2)
        self.assertEqual(len(top), 2)
        # Top companion is FastAPI with count 2
        self.assertEqual(top[0], ("FastAPI", 2))
        # Second companion has count 1 (Docker or React)
        self.assertEqual(top[1][1], 1)

    def test_incremental_update(self):
        # Start with 2 postings
        model = CooccurrenceMatrix().fit([
            ["Python", "Docker"],
        ])
        self.assertEqual(model.get_count("Python", "Docker"), 1)

        # Incrementally add more postings
        model.update([
            ["Python", "Docker"],
            ["Docker", "Kubernetes"],
        ])
        self.assertEqual(model.get_count("Python", "Docker"), 2)
        self.assertEqual(model.get_count("Docker", "Kubernetes"), 1)


# ---------------------------------------------------------------------------
# Test Suite 6: Serialization & Export Formats
# ---------------------------------------------------------------------------

class TestSerialization(unittest.TestCase):
    """Tests exporting to nested dict, edge list, CSV, and JSON."""

    def setUp(self):
        self.model = CooccurrenceMatrix().fit(SAMPLE_POSTINGS)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_to_dict_format(self):
        d = self.model.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("Python", d)
        self.assertEqual(d["Python"]["FastAPI"], 2)

    def test_to_edges_format_and_sorting(self):
        edges = self.model.to_edges()
        self.assertIsInstance(edges, list)
        self.assertGreater(len(edges), 0)

        # Top edge should have the highest weight
        self.assertEqual(edges[0]["weight"], 2)
        self.assertEqual(set([edges[0]["source"], edges[0]["target"]]), {"FastAPI", "Python"})

        # Check edge structure
        for edge in edges:
            self.assertIn("source", edge)
            self.assertIn("target", edge)
            self.assertIn("weight", edge)
            self.assertIsInstance(edge["weight"], int)

    def test_to_csv_roundtrip(self):
        csv_path = Path(self.tmp_dir) / "matrix.csv"
        saved = self.model.to_csv(csv_path)
        self.assertTrue(saved.exists())

        # Read back with pandas
        loaded_df = pd.read_csv(saved, index_col=0)
        pd.testing.assert_frame_equal(self.model.matrix, loaded_df, check_dtype=False)

    def test_to_json_roundtrip(self):
        json_path = Path(self.tmp_dir) / "matrix.json"
        saved = self.model.to_json(json_path)
        self.assertTrue(saved.exists())

        with open(saved, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["Python"]["FastAPI"], 2)
        self.assertEqual(data["Python"]["Python"], 0)


# ---------------------------------------------------------------------------
# Test Suite 7: Module-Level Convenience Functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions(unittest.TestCase):
    """Tests build_cooccurrence_matrix and build_network_edges."""

    def test_build_cooccurrence_matrix(self):
        df = build_cooccurrence_matrix(SAMPLE_POSTINGS)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.loc["Python", "FastAPI"], 2)
        self.assertEqual(df.loc["Python", "Python"], 0)

    def test_build_network_edges(self):
        edges = build_network_edges(SAMPLE_POSTINGS, min_weight=1)
        self.assertIsInstance(edges, list)
        self.assertGreaterEqual(len(edges), 3)


if __name__ == "__main__":
    unittest.main()
