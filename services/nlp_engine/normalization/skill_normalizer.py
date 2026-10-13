"""
skill_normalizer.py — Skill Normalization & Canonical Resolution Engine
=======================================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  1. Map raw surface forms (e.g. "React.js", "python3", "Docker Compose")
     to their canonical taxonomy names ("React", "Python", "Docker").
  2. Assign confidence scores:
       - Exact / alias / taxonomy_id match : 1.0 (RULE_BASED_CONFIDENCE)
       - Fuzzy fallback (difflib)          : similarity ratio (e.g. 0.85 - 0.99)
  3. Filter out unknown terms (< MIN_CONFIDENCE_THRESHOLD or unmapped).
  4. Deduplicate multiple variants of the same canonical skill per job posting.
  5. Produce both rich metadata (NormalizedSkill) and simple canonical string
     lists ready for Contract A JSON output.

Cross-team compatibility notes:
  - canonical_name is the shared contract vocabulary with Member 3 (forecasting)
    and Member 4 (dashboard visualization).
  - ZERO external dependencies outside Python stdlib (difflib, logging, typing)
    and internal config/taxonomy modules.
"""

import difflib
import logging
from typing import Any, Optional, Union

from services.nlp_engine.config import (
    LOG_LEVEL,
    MIN_CONFIDENCE_THRESHOLD,
    RULE_BASED_CONFIDENCE,
)
from services.nlp_engine.taxonomy.taxonomy_loader import (
    TaxonomyEntry,
    load_taxonomy,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type Definitions
# ---------------------------------------------------------------------------

# Input representation: can be raw string or a SkillHit dict from NERPipeline
SkillInput = Union[str, dict]

# Structured output representation of a normalized skill
NormalizedSkill = dict[str, Any]
# Schema:
# {
#     "canonical_name": str,     # Canonical skill name ("React")
#     "category"      : str,     # Category ("Frontend")
#     "taxonomy_id"   : str,     # Taxonomy ID ("t-101")
#     "raw_mention"   : str,     # Original surface text ("React.js")
#     "confidence"    : float,   # Score between 0.0 and 1.0
#     "method"        : str,     # Resolution method: "taxonomy_id" | "exact_alias" | "fuzzy"
# }


# ---------------------------------------------------------------------------
# SkillNormalizer Class
# ---------------------------------------------------------------------------

class SkillNormalizer:
    """
    Normalizes raw skill mentions to canonical taxonomy entries.

    Features:
      - Taxonomy ID resolution (fastest, deterministic for NER hits).
      - Exact alias resolution (case-insensitive, whitespace-trimmed).
      - Pure standard-library fuzzy fallback (difflib) for typos/minor variants.
      - Per-document deduplication by canonical name.
      - Produces clean string lists for Contract A.

    Supports dependency injection of custom taxonomy entries for isolated testing.
    """

    def __init__(
        self,
        taxonomy_entries: Optional[list[TaxonomyEntry]] = None,
        fuzzy_threshold: float = 0.85,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        Initializes the normalizer with lookup tables.

        Args:
            taxonomy_entries: Optional list of taxonomy dicts. If None, loads
                              from the default taxonomy file.
            fuzzy_threshold : Minimum difflib similarity ratio to accept a fuzzy match.
                              Default 0.85.
            min_confidence  : Absolute minimum confidence cutoff. Matches below this
                              are discarded. Default from config (0.5).
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.min_confidence = min_confidence

        entries = taxonomy_entries if taxonomy_entries is not None else load_taxonomy()
        self._entries: list[TaxonomyEntry] = entries

        # Build in-memory lookup indexes
        self._alias_to_canonical: dict[str, str] = {}
        self._id_to_entry: dict[str, TaxonomyEntry] = {}
        self._canonical_to_entry: dict[str, TaxonomyEntry] = {}
        self._all_lookup_keys: list[str] = []

        self._build_indexes(entries)

    def _build_indexes(self, entries: list[TaxonomyEntry]) -> None:
        """Constructs fast reverse-lookup dictionaries from taxonomy entries."""
        for entry in entries:
            canonical = entry["canonical_name"]
            tax_id = entry.get("id", canonical)

            self._id_to_entry[tax_id] = entry
            self._canonical_to_entry[canonical] = entry

            # Canonical itself is an alias
            self._alias_to_canonical[canonical.lower()] = canonical

            for alias in entry.get("aliases", []):
                alias_clean = alias.strip().lower()
                if alias_clean:
                    if alias_clean not in self._alias_to_canonical:
                        self._alias_to_canonical[alias_clean] = canonical

        # Pre-cache list of all lowercase aliases for difflib matching
        self._all_lookup_keys = list(self._alias_to_canonical.keys())

    # -----------------------------------------------------------------------
    # Core Resolution Logic
    # -----------------------------------------------------------------------

    def normalize(
        self,
        skill_input: SkillInput,
        allow_fuzzy: bool = False,
    ) -> Optional[NormalizedSkill]:
        """
        Normalizes a single skill mention or SkillHit dict to a NormalizedSkill.

        Resolution hierarchy:
          1. By `taxonomy_id` (if skill_input is a dict with valid `taxonomy_id`)
          2. By exact surface form / alias lookup (case-insensitive)
          3. By fuzzy similarity (if allow_fuzzy=True and score >= fuzzy_threshold)

        Args:
            skill_input: Raw skill string (e.g. "React.js") or hit dict
                         (e.g. {"text": "React.js", "taxonomy_id": "t-101"}).
            allow_fuzzy: If True, enables difflib fuzzy matching fallback.

        Returns:
            NormalizedSkill dict if successfully resolved, or None if unknown/rejected.
        """
        if skill_input is None:
            return None

        raw_text: str = ""
        provided_id: Optional[str] = None

        if isinstance(skill_input, dict):
            raw_text = str(skill_input.get("text", "")).strip()
            provided_id = skill_input.get("taxonomy_id")
        elif isinstance(skill_input, str):
            raw_text = skill_input.strip()
        else:
            try:
                raw_text = str(skill_input).strip()
            except Exception:
                return None

        if not raw_text and not provided_id:
            return None

        # 1. Resolve by taxonomy_id if provided and exists
        if provided_id and provided_id in self._id_to_entry:
            entry = self._id_to_entry[provided_id]
            return {
                "canonical_name": entry["canonical_name"],
                "category": entry.get("category", "General"),
                "taxonomy_id": provided_id,
                "raw_mention": raw_text or entry["canonical_name"],
                "confidence": RULE_BASED_CONFIDENCE,
                "method": "taxonomy_id",
            }

        # 2. Resolve by exact alias lookup (case-insensitive)
        text_lower = raw_text.lower()
        if text_lower in self._alias_to_canonical:
            canonical = self._alias_to_canonical[text_lower]
            entry = self._canonical_to_entry[canonical]
            return {
                "canonical_name": canonical,
                "category": entry.get("category", "General"),
                "taxonomy_id": entry.get("id", canonical),
                "raw_mention": raw_text,
                "confidence": RULE_BASED_CONFIDENCE,
                "method": "exact_alias",
            }

        # 3. Optional fuzzy fallback using standard library difflib
        if allow_fuzzy and text_lower:
            closest_matches = difflib.get_close_matches(
                text_lower,
                self._all_lookup_keys,
                n=1,
                cutoff=self.fuzzy_threshold,
            )
            if closest_matches:
                best_alias = closest_matches[0]
                ratio = difflib.SequenceMatcher(None, text_lower, best_alias).ratio()
                confidence = round(ratio, 4)

                if confidence >= self.min_confidence:
                    canonical = self._alias_to_canonical[best_alias]
                    entry = self._canonical_to_entry[canonical]
                    logger.debug(
                        f"[skill_normalizer] Fuzzy match: '{raw_text}' -> '{canonical}' "
                        f"(via '{best_alias}', confidence={confidence})"
                    )
                    return {
                        "canonical_name": canonical,
                        "category": entry.get("category", "General"),
                        "taxonomy_id": entry.get("id", canonical),
                        "raw_mention": raw_text,
                        "confidence": confidence,
                        "method": "fuzzy",
                    }

        logger.debug(f"[skill_normalizer] Unrecognized skill: '{raw_text}'")
        return None

    # -----------------------------------------------------------------------
    # List & Batch Normalization
    # -----------------------------------------------------------------------

    def normalize_list(
        self,
        skills: list[SkillInput],
        deduplicate: bool = True,
        allow_fuzzy: bool = False,
    ) -> list[NormalizedSkill]:
        """
        Normalizes a list of skill mentions/hits for a single job posting.

        Args:
            skills     : List of skill strings or SkillHit dicts.
            deduplicate: If True, deduplicates hits by canonical_name, keeping
                         the entry with the highest confidence score.
            allow_fuzzy: If True, enables fuzzy fallback for unmatched terms.

        Returns:
            List of NormalizedSkill dicts.
        """
        if not skills:
            return []

        resolved: list[NormalizedSkill] = []
        for item in skills:
            norm = self.normalize(item, allow_fuzzy=allow_fuzzy)
            if norm is not None:
                resolved.append(norm)

        if not deduplicate:
            return resolved

        # Deduplicate by canonical_name, retaining the highest confidence
        dedup_map: dict[str, NormalizedSkill] = {}
        for item in resolved:
            canonical = item["canonical_name"]
            if canonical not in dedup_map:
                dedup_map[canonical] = item
            else:
                if item["confidence"] > dedup_map[canonical]["confidence"]:
                    dedup_map[canonical] = item

        return list(dedup_map.values())

    def to_canonical_list(
        self,
        skills: list[SkillInput],
        deduplicate: bool = True,
        sort_result: bool = True,
        allow_fuzzy: bool = False,
    ) -> list[str]:
        """
        Convenience method to return a clean list of canonical skill names (strings).
        Directly powers the `extracted_skills` array for Contract A JSON output.

        Args:
            skills     : List of raw skill strings or hit dicts.
            deduplicate: If True (default), removes duplicate canonical skills.
            sort_result: If True (default), sorts alphabetically for determinism.
            allow_fuzzy: If True, enables fuzzy matching.

        Returns:
            List of canonical skill name strings, e.g. ["Docker", "FastAPI", "Python"].
        """
        norm_list = self.normalize_list(
            skills,
            deduplicate=deduplicate,
            allow_fuzzy=allow_fuzzy,
        )
        canonical_names = [item["canonical_name"] for item in norm_list]

        if deduplicate:
            # Preserve uniqueness if deduplicate was somehow bypassed
            canonical_names = list(dict.fromkeys(canonical_names))

        if sort_result:
            canonical_names.sort()

        return canonical_names

    def normalize_batch(
        self,
        batch: list[list[SkillInput]],
        deduplicate: bool = True,
        allow_fuzzy: bool = False,
    ) -> list[list[NormalizedSkill]]:
        """
        Normalizes a batch of skill lists (e.g. from multiple job descriptions).

        Args:
            batch      : List of skill lists (one per job description).
            deduplicate: Whether to deduplicate canonical skills within each job.
            allow_fuzzy: Whether to allow fuzzy matching.

        Returns:
            List of normalized skill lists, aligned with input indices.
        """
        return [
            self.normalize_list(job_skills, deduplicate=deduplicate, allow_fuzzy=allow_fuzzy)
            for job_skills in batch
        ]

    def batch_to_canonical_lists(
        self,
        batch: list[list[SkillInput]],
        deduplicate: bool = True,
        sort_result: bool = True,
        allow_fuzzy: bool = False,
    ) -> list[list[str]]:
        """
        Normalizes a batch of skill lists directly into lists of canonical name strings.

        Args:
            batch      : List of skill lists.
            deduplicate: Whether to deduplicate within each job posting.
            sort_result: Whether to sort names alphabetically.
            allow_fuzzy: Whether to allow fuzzy matching.

        Returns:
            List of canonical name string lists.
        """
        return [
            self.to_canonical_list(
                job_skills,
                deduplicate=deduplicate,
                sort_result=sort_result,
                allow_fuzzy=allow_fuzzy,
            )
            for job_skills in batch
        ]


# ---------------------------------------------------------------------------
# Module-level Singleton & Convenience Functions
# ---------------------------------------------------------------------------

_default_normalizer: Optional[SkillNormalizer] = None


def get_default_normalizer() -> SkillNormalizer:
    """Returns the lazy-loaded default SkillNormalizer singleton."""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = SkillNormalizer()
    return _default_normalizer


def normalize(
    skill_input: SkillInput,
    allow_fuzzy: bool = False,
) -> Optional[NormalizedSkill]:
    """Module-level convenience wrapper for normalize()."""
    return get_default_normalizer().normalize(skill_input, allow_fuzzy=allow_fuzzy)


def normalize_list(
    skills: list[SkillInput],
    deduplicate: bool = True,
    allow_fuzzy: bool = False,
) -> list[NormalizedSkill]:
    """Module-level convenience wrapper for normalize_list()."""
    return get_default_normalizer().normalize_list(
        skills, deduplicate=deduplicate, allow_fuzzy=allow_fuzzy
    )


def to_canonical_list(
    skills: list[SkillInput],
    deduplicate: bool = True,
    sort_result: bool = True,
    allow_fuzzy: bool = False,
) -> list[str]:
    """Module-level convenience wrapper for to_canonical_list()."""
    return get_default_normalizer().to_canonical_list(
        skills,
        deduplicate=deduplicate,
        sort_result=sort_result,
        allow_fuzzy=allow_fuzzy,
    )
