"""
cooccurrence_matrix.py — Skill Co-occurrence Matrix Engine
===========================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  1. Generate unique pairwise skill combinations (itertools.combinations)
     for each job posting's extracted skill list.
  2. Build a symmetric co-occurrence frequency matrix (pandas.DataFrame)
     using collections.Counter for fast frequency aggregation.
  3. Ensure diagonal values are strictly 0 (skills do not co-occur with themselves).
  4. Serialize results to CSV, JSON, nested dict, or graph edges list
     for Member 3 (Prophet/Forecasting) and Member 4 (React Flow Graph).

Decoupling / Cross-team notes:
  - Input: List of canonical skill name lists (from Phase 4 skill_normalizer).
  - Output formats:
      - DataFrame / CSV / JSON (Contract with Member 3)
      - Edge list [{"source": s1, "target": s2, "weight": count}] (for Member 4)
  - Zero external dependencies beyond stdlib and pandas.
"""

from collections import Counter
import itertools
import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from services.nlp_engine.config import (
    LOG_LEVEL,
    MIN_COOCCURRENCE_COUNT,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

SkillList = list[str]
PostingsSkills = list[SkillList]
PairTuple = tuple[str, str]
NetworkEdge = dict[str, Any]  # {"source": str, "target": str, "weight": int}


# ---------------------------------------------------------------------------
# CooccurrenceMatrix Class
# ---------------------------------------------------------------------------

class CooccurrenceMatrix:
    """
    Computes, maintains, and exports a symmetric skill co-occurrence matrix.

    Usage:
        model = CooccurrenceMatrix()
        model.fit([
            ["Python", "FastAPI", "Docker"],
            ["Python", "React"],
            ["Python", "FastAPI"],
        ])
        df = model.matrix
        edges = model.to_edges()
    """

    def __init__(
        self,
        min_count: int = MIN_COOCCURRENCE_COUNT,
        vocabulary: Optional[list[str]] = None,
    ) -> None:
        """
        Initializes the co-occurrence matrix builder.

        Args:
            min_count : Minimum co-occurrence count threshold. Pairs with count
                        less than this will be set to 0. Default from config (1).
            vocabulary: Optional predefined list of canonical skill names to fix
                        matrix dimensions. If None, vocabulary is derived from
                        all unique skills observed in the input data.
        """
        self.min_count = max(1, min_count)
        self._fixed_vocabulary = sorted(list(dict.fromkeys(vocabulary))) if vocabulary else None
        self._fixed_vocabulary_set = set(self._fixed_vocabulary) if self._fixed_vocabulary else None
        self._pair_counts: Counter[PairTuple] = Counter()
        self._unique_skills: set[str] = set(self._fixed_vocabulary or [])
        self._df_cache: Optional[pd.DataFrame] = None

    # -----------------------------------------------------------------------
    # Core Processing
    # -----------------------------------------------------------------------

    def fit(self, postings: PostingsSkills) -> "CooccurrenceMatrix":
        """
        Resets and populates the co-occurrence counts from a list of job postings.

        Args:
            postings: List of skill lists (one list of canonical skill strings per job).

        Returns:
            self for method chaining.
        """
        self._pair_counts.clear()
        self._unique_skills = set(self._fixed_vocabulary or [])
        self._df_cache = None

        self.update(postings)
        return self

    def update(self, postings: PostingsSkills) -> "CooccurrenceMatrix":
        """
        Incrementally adds new job postings to existing co-occurrence counts.

        Args:
            postings: Additional list of skill lists.

        Returns:
            self for method chaining.
        """
        if not postings:
            return self

        self._df_cache = None  # Invalidate cached DataFrame

        for job_skills in postings:
            if not job_skills:
                continue

            # Clean and deduplicate skills within this single job
            cleaned_skills = {
                str(s).strip()
                for s in job_skills
                if s is not None and str(s).strip()
            }

            # If fixed vocabulary is set, filter out skills outside vocabulary
            if self._fixed_vocabulary_set is not None:
                cleaned_skills = {s for s in cleaned_skills if s in self._fixed_vocabulary_set}

            unique_job_skills = sorted(list(cleaned_skills))

            # Record skills observed
            for skill in unique_job_skills:
                self._unique_skills.add(skill)

            # Generate all unique 2-combinations (sorted order guarantees consistency)
            for s1, s2 in itertools.combinations(unique_job_skills, 2):
                pair = (s1, s2) if s1 < s2 else (s2, s1)
                self._pair_counts[pair] += 1

        logger.debug(
            f"[cooccurrence] Updated matrix with {len(postings)} postings. "
            f"Total unique skills: {len(self._unique_skills)}, "
            f"Unique pairs: {len(self._pair_counts)}"
        )
        return self

    # -----------------------------------------------------------------------
    # Matrix Generation
    # -----------------------------------------------------------------------

    @property
    def matrix(self) -> pd.DataFrame:
        """
        Constructs and returns the symmetric co-occurrence DataFrame.
        Cached until new data is added.

        Returns:
            Symmetric pd.DataFrame with sorted skills on rows and columns,
            and diagonal set to 0.
        """
        if self._df_cache is not None:
            return self._df_cache

        skills = self._fixed_vocabulary if self._fixed_vocabulary is not None else sorted(list(self._unique_skills))
        n = len(skills)

        if n == 0:
            self._df_cache = pd.DataFrame(dtype="int64")
            return self._df_cache

        # Create zero-initialized DataFrame with int64
        df = pd.DataFrame(0, index=skills, columns=skills, dtype="int64")

        # Fill symmetric pairs that meet min_count
        for (s1, s2), count in self._pair_counts.items():
            if count >= self.min_count:
                if s1 in df.index and s2 in df.columns:
                    df.at[s1, s2] = count
                    df.at[s2, s1] = count

        # Explicitly ensure diagonal is 0
        for skill in skills:
            df.at[skill, skill] = 0

        self._df_cache = df
        return self._df_cache

    @property
    def pairs_counter(self) -> Counter[PairTuple]:
        """Returns the internal Counter mapping (skill_a, skill_b) -> frequency."""
        return self._pair_counts

    @property
    def vocabulary(self) -> list[str]:
        """Returns the sorted list of skills tracked by this matrix."""
        return sorted(list(self._unique_skills))

    # -----------------------------------------------------------------------
    # Helper & Query Methods
    # -----------------------------------------------------------------------

    def get_count(self, skill_a: str, skill_b: str) -> int:
        """
        Returns the co-occurrence count between two skills.
        Returns 0 if skills are identical, unknown, or below min_count.
        """
        if not skill_a or not skill_b or skill_a == skill_b:
            return 0

        pair = (skill_a, skill_b) if skill_a < skill_b else (skill_b, skill_a)
        count = self._pair_counts.get(pair, 0)
        return count if count >= self.min_count else 0

    def get_top_cooccurring(
        self,
        skill: str,
        top_n: int = 5,
    ) -> list[tuple[str, int]]:
        """
        Finds the top N skills that most frequently co-occur with the given skill.

        Args:
            skill: Canonical skill name.
            top_n: Number of top results to return.

        Returns:
            List of (companion_skill, frequency) tuples sorted by frequency desc.
        """
        df = self.matrix
        if skill not in df.index:
            return []

        row = df.loc[skill]
        # Drop self (0) and filter for count > 0
        non_zero = row[row > 0].sort_values(ascending=False)
        top_series = non_zero.head(top_n)
        return [(str(companion), int(count)) for companion, count in top_series.items()]

    # -----------------------------------------------------------------------
    # Serialization & Exports
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict[str, dict[str, int]]:
        """
        Serializes matrix to a nested dictionary for JSON export.
        Shape: { "Python": { "FastAPI": 45, "Docker": 30, ... }, ... }
        """
        return self.matrix.to_dict(orient="index")

    def to_edges(self, min_weight: Optional[int] = None) -> list[NetworkEdge]:
        """
        Exports co-occurrence pairs as an edge list for network graph visualization
        (Member 4's React Flow / co-occurrence network component).

        Args:
            min_weight: Optional filter; if None, uses self.min_count.

        Returns:
            List of edge dicts: [{"source": "Docker", "target": "Python", "weight": 30}, ...]
        """
        threshold = self.min_count if min_weight is None else max(1, min_weight)
        edges: list[NetworkEdge] = []

        for (s1, s2), count in sorted(self._pair_counts.items(), key=lambda x: -x[1]):
            if count >= threshold:
                edges.append({
                    "source": s1,
                    "target": s2,
                    "weight": count,
                })
        return edges

    def to_csv(self, filepath: Union[str, Path]) -> Path:
        """
        Saves the matrix to a CSV file.

        Args:
            filepath: Destination path.

        Returns:
            Resolved Path of the saved file.
        """
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.matrix.to_csv(path, index=True)
        logger.info(f"[cooccurrence] Matrix saved to CSV: {path}")
        return path

    def to_json(self, filepath: Union[str, Path], orient: str = "index") -> Path:
        """
        Saves the matrix to a JSON file.

        Args:
            filepath: Destination path.
            orient  : Pandas orient string ("index", "split", etc.). Default "index".

        Returns:
            Resolved Path of the saved file.
        """
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.matrix.to_json(path, orient=orient, indent=2)
        logger.info(f"[cooccurrence] Matrix saved to JSON: {path}")
        return path


# ---------------------------------------------------------------------------
# Module-level Convenience Functions
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(
    postings: PostingsSkills,
    min_count: int = MIN_COOCCURRENCE_COUNT,
    vocabulary: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convenience function to compute and return a co-occurrence DataFrame in one call.

    Args:
        postings  : List of skill lists per job posting.
        min_count : Minimum co-occurrence threshold.
        vocabulary: Optional fixed vocabulary list.

    Returns:
        Symmetric pandas DataFrame with 0 on diagonal.
    """
    model = CooccurrenceMatrix(min_count=min_count, vocabulary=vocabulary)
    model.fit(postings)
    return model.matrix


def build_network_edges(
    postings: PostingsSkills,
    min_weight: int = MIN_COOCCURRENCE_COUNT,
) -> list[NetworkEdge]:
    """
    Convenience function to generate network graph edges directly from job postings.

    Args:
        postings  : List of skill lists.
        min_weight: Minimum co-occurrence weight.

    Returns:
        List of edge dicts ready for frontend graph visualization.
    """
    model = CooccurrenceMatrix(min_count=min_weight)
    model.fit(postings)
    return model.to_edges(min_weight=min_weight)
