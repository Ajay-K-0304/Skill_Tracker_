"""
config.py — Central Configuration for the NLP Engine Module
============================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology
Directory: services/nlp_engine/

ALL configurable values for the entire nlp_engine package live here.
No other module should hardcode paths, model names, or thresholds.
Import this module to access any setting.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------

# Root of the nlp_engine package (this file's parent directory)
NLP_ENGINE_DIR: Path = Path(__file__).resolve().parent

# Taxonomy data directory
TAXONOMY_DIR: Path = NLP_ENGINE_DIR / "taxonomy"

# Tests directory
TESTS_DIR: Path = NLP_ENGINE_DIR / "tests"

# ---------------------------------------------------------------------------
# Taxonomy Settings
# ---------------------------------------------------------------------------

# Path to the master skill ontology JSON file
TAXONOMY_FILE: Path = TAXONOMY_DIR / "skill_taxonomy.json"

# Expected keys in every taxonomy entry (used for validation)
TAXONOMY_REQUIRED_KEYS: tuple = ("canonical_name", "category", "aliases")

# ---------------------------------------------------------------------------
# spaCy Model Settings
# ---------------------------------------------------------------------------

# spaCy model to load. Use "en_core_web_sm" for lightweight / CI,
# switch to "en_core_web_md" for better accuracy in production.
# Override via environment variable: NLP_SPACY_MODEL=en_core_web_md
SPACY_MODEL: str = os.environ.get("NLP_SPACY_MODEL", "en_core_web_sm")

# Custom EntityRuler label used to tag matched skills in spaCy docs
SKILL_ENTITY_LABEL: str = "SKILL"

# Pipeline components to DISABLE from the base spaCy model for speed.
# We only need the tokenizer + our EntityRuler; no need for parser or NER.
SPACY_DISABLED_COMPONENTS: list = ["ner", "parser", "lemmatizer"]

# ---------------------------------------------------------------------------
# Skill Extraction Settings
# ---------------------------------------------------------------------------

# Confidence score assigned to rule-based exact/alias matches.
# (Fuzzy or statistical matches will use lower scores in future phases.)
RULE_BASED_CONFIDENCE: float = 1.0

# Minimum confidence threshold — skills below this are discarded.
MIN_CONFIDENCE_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Co-occurrence Matrix Settings
# ---------------------------------------------------------------------------

# Minimum number of times a skill pair must co-occur to be included
# in the output matrix (filters out noise from rare pairings).
MIN_COOCCURRENCE_COUNT: int = 1

# ---------------------------------------------------------------------------
# Output / Contract Settings
# ---------------------------------------------------------------------------

# Date format used in Contract A JSON output (ISO 8601)
DATE_FORMAT: str = "%Y-%m-%d"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Log level for the nlp_engine package.
# Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL: str = os.environ.get("NLP_LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Sanity Check (runs on import — catches misconfigured paths early)
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    """
    Validates that critical paths exist.
    Raises FileNotFoundError with a descriptive message if anything is wrong.
    Called automatically when this module is imported.
    """
    if not NLP_ENGINE_DIR.exists():
        raise FileNotFoundError(
            f"[config] NLP_ENGINE_DIR not found: {NLP_ENGINE_DIR}"
        )
    if not TAXONOMY_DIR.exists():
        raise FileNotFoundError(
            f"[config] TAXONOMY_DIR not found: {TAXONOMY_DIR}\n"
            "Ensure the 'taxonomy/' folder exists inside nlp_engine/."
        )
    # TAXONOMY_FILE is allowed to not exist yet (Phase 1 creates it)


_validate_config()
