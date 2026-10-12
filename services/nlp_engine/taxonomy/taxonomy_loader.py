"""
taxonomy_loader.py — Skill Taxonomy Loader & Lookup Engine
===========================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  1. Load and validate skill_taxonomy.json from disk.
  2. Build two fast in-memory lookup structures:
       - alias_to_canonical  : lowercase alias → canonical_name
       - canonical_to_entry  : canonical_name  → full taxonomy entry dict
  3. Expose a clean, stateless public API consumed by:
       - pipeline/entity_ruler.py   (spaCy pattern building)
       - normalization/skill_normalizer.py (alias → canonical mapping)

Cross-team compatibility notes:
  - canonical_name values are the SHARED vocabulary across ALL members.
    Member 3 forecasts by canonical_name; Member 4 displays canonical_name.
    DO NOT change canonical_name values without team-wide sync.
  - Entry structure mirrors the PostgreSQL `skill_taxonomy` table:
      id (VARCHAR), canonical_name (VARCHAR), category (VARCHAR), aliases (TEXT[])
  - This module has ZERO imports outside Python stdlib + config.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from services.nlp_engine.config import (
    TAXONOMY_FILE,
    TAXONOMY_REQUIRED_KEYS,
    LOG_LEVEL,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types (plain dicts — no external dependencies)
# ---------------------------------------------------------------------------

# A single taxonomy entry as loaded from JSON
TaxonomyEntry = dict  # keys: id, canonical_name, category, aliases

# ---------------------------------------------------------------------------
# Internal State (module-level singletons, loaded once)
# ---------------------------------------------------------------------------

_taxonomy: list[TaxonomyEntry] = []          # raw list of all entries
_alias_to_canonical: dict[str, str] = {}     # lowercase alias → canonical_name
_canonical_to_entry: dict[str, TaxonomyEntry] = {}  # canonical_name → full entry
_loaded: bool = False                         # guard against double-loading


# ---------------------------------------------------------------------------
# Private Helpers
# ---------------------------------------------------------------------------

def _validate_entry(entry: dict, index: int) -> bool:
    """
    Validates a single taxonomy entry has all required keys and correct types.

    Args:
        entry: The dict to validate.
        index: Position in the JSON array (used in error messages).

    Returns:
        True if valid, False otherwise (logs a warning on failure).
    """
    for key in TAXONOMY_REQUIRED_KEYS:
        if key not in entry:
            logger.warning(
                f"[taxonomy_loader] Entry[{index}] missing required key '{key}'. "
                f"Entry skipped: {entry}"
            )
            return False

    if not isinstance(entry["canonical_name"], str) or not entry["canonical_name"].strip():
        logger.warning(
            f"[taxonomy_loader] Entry[{index}] 'canonical_name' must be a non-empty string. "
            f"Entry skipped: {entry}"
        )
        return False

    if not isinstance(entry["aliases"], list):
        logger.warning(
            f"[taxonomy_loader] Entry[{index}] 'aliases' must be a list. "
            f"Entry skipped: {entry}"
        )
        return False

    return True


def _build_lookup_tables(entries: list[TaxonomyEntry]) -> tuple[dict, dict]:
    """
    Builds alias_to_canonical and canonical_to_entry dicts from validated entries.

    Alias matching is case-insensitive: all alias keys are stored lowercase.
    The canonical_name itself is also registered as an alias of itself.

    Args:
        entries: List of validated taxonomy entry dicts.

    Returns:
        Tuple of (alias_to_canonical, canonical_to_entry).
    """
    alias_to_canonical: dict[str, str] = {}
    canonical_to_entry: dict[str, TaxonomyEntry] = {}

    for entry in entries:
        canonical = entry["canonical_name"]
        canonical_to_entry[canonical] = entry

        # Register the canonical name itself (lowercase) as a self-alias
        alias_to_canonical[canonical.lower()] = canonical

        # Register every alias (lowercase) → canonical
        for alias in entry["aliases"]:
            alias_lower = alias.strip().lower()
            if alias_lower in alias_to_canonical:
                existing = alias_to_canonical[alias_lower]
                if existing != canonical:
                    logger.warning(
                        f"[taxonomy_loader] Alias conflict: '{alias}' already mapped to "
                        f"'{existing}', skipping mapping to '{canonical}'."
                    )
                continue
            alias_to_canonical[alias_lower] = canonical

    return alias_to_canonical, canonical_to_entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_taxonomy(filepath: Path = TAXONOMY_FILE) -> list[TaxonomyEntry]:
    """
    Loads, validates, and indexes the skill taxonomy from disk.
    Idempotent — safe to call multiple times; reloads only on first call.

    Args:
        filepath: Path to the taxonomy JSON file. Defaults to TAXONOMY_FILE
                  from config. Override only for testing.

    Returns:
        List of validated taxonomy entry dicts.

    Raises:
        FileNotFoundError : If the JSON file does not exist at filepath.
        ValueError        : If the JSON file is empty or not a list.
    """
    global _taxonomy, _alias_to_canonical, _canonical_to_entry, _loaded

    if _loaded:
        logger.debug("[taxonomy_loader] Taxonomy already loaded. Returning cached data.")
        return _taxonomy

    # --- Load ---
    if not filepath.exists():
        raise FileNotFoundError(
            f"[taxonomy_loader] Taxonomy file not found: {filepath}\n"
            "Run Phase 1 setup to create skill_taxonomy.json."
        )

    with open(filepath, encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list) or len(raw_data) == 0:
        raise ValueError(
            f"[taxonomy_loader] Taxonomy file must be a non-empty JSON array. "
            f"Got: {type(raw_data).__name__}"
        )

    # --- Validate entries ---
    valid_entries: list[TaxonomyEntry] = []
    for i, entry in enumerate(raw_data):
        if _validate_entry(entry, i):
            valid_entries.append(entry)

    skipped = len(raw_data) - len(valid_entries)
    if skipped > 0:
        logger.warning(f"[taxonomy_loader] {skipped} invalid entries skipped.")

    # --- Build lookup tables ---
    _alias_to_canonical, _canonical_to_entry = _build_lookup_tables(valid_entries)
    _taxonomy = valid_entries
    _loaded = True

    logger.info(
        f"[taxonomy_loader] Loaded {len(_taxonomy)} skills | "
        f"{len(_alias_to_canonical)} aliases indexed."
    )
    return _taxonomy


def get_alias_to_canonical() -> dict[str, str]:
    """
    Returns the alias→canonical lookup dict (lowercase keys).
    Loads taxonomy on first call if not already loaded.

    Returns:
        Dict mapping every lowercase alias (and canonical name itself) to its
        canonical_name string. Example: {"react.js": "React", "reactjs": "React"}
    """
    if not _loaded:
        load_taxonomy()
    return _alias_to_canonical


def get_canonical_to_entry() -> dict[str, TaxonomyEntry]:
    """
    Returns the canonical_name → full entry dict.
    Loads taxonomy on first call if not already loaded.

    Returns:
        Dict mapping canonical_name to its full taxonomy entry dict.
    """
    if not _loaded:
        load_taxonomy()
    return _canonical_to_entry


def get_all_canonical_names() -> list[str]:
    """
    Returns a sorted list of all canonical skill names.
    Useful for Member 3 (forecasting keys) and Member 4 (display labels).

    Returns:
        Sorted list of canonical_name strings.
    """
    if not _loaded:
        load_taxonomy()
    return sorted(_canonical_to_entry.keys())


def resolve_alias(raw_skill: str) -> Optional[str]:
    """
    Resolves a raw/variant skill string to its canonical name.
    Lookup is case-insensitive.

    Args:
        raw_skill: Raw skill text (e.g. "React.js", "ReactJS", "reactjs").

    Returns:
        Canonical name string if found (e.g. "React"), or None if not in taxonomy.

    Example:
        >>> resolve_alias("React.js")
        'React'
        >>> resolve_alias("unknown_tool_xyz")
        None
    """
    if not _loaded:
        load_taxonomy()
    return _alias_to_canonical.get(raw_skill.strip().lower())


def get_categories() -> list[str]:
    """
    Returns a sorted list of all unique skill categories in the taxonomy.
    Useful for Member 4 dashboard filtering.

    Returns:
        Sorted list of unique category strings.
    """
    if not _loaded:
        load_taxonomy()
    return sorted({entry["category"] for entry in _taxonomy})


def reload_taxonomy(filepath: Path = TAXONOMY_FILE) -> list[TaxonomyEntry]:
    """
    Forces a fresh reload of the taxonomy (clears cached state).
    Use only in tests or when taxonomy file changes at runtime.

    Args:
        filepath: Path to the taxonomy JSON file.

    Returns:
        Freshly loaded list of taxonomy entry dicts.
    """
    global _loaded
    _loaded = False
    logger.info("[taxonomy_loader] Cache cleared. Reloading taxonomy...")
    return load_taxonomy(filepath)
