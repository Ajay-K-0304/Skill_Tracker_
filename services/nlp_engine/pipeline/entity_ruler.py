"""
entity_ruler.py — spaCy EntityRuler Builder
=============================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  Converts the skill taxonomy into spaCy EntityRuler phrase patterns and
  attaches the ruler component to a given spaCy nlp pipeline.

Why EntityRuler (not statistical NER)?
  - Rule-based: deterministic, no retraining needed.
  - Phrase patterns match exact aliases (e.g. "React.js", "ReactJS").
  - `phrase_matcher_attr="LOWER"` → case-insensitive matching for free.
  - The `id` field on each pattern links back to the taxonomy entry ID,
    available at runtime via `ent.ent_id_` on any matched span.

This module has ZERO imports outside Python stdlib + spaCy + config/taxonomy.
"""

import logging
from typing import Any

import spacy
from spacy.language import Language

from services.nlp_engine.config import (
    LOG_LEVEL,
    SKILL_ENTITY_LABEL,
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
# Pattern Builder
# ---------------------------------------------------------------------------

def build_patterns(taxonomy_entries: list[TaxonomyEntry]) -> list[dict]:
    """
    Converts taxonomy entries into a flat list of spaCy EntityRuler patterns.

    Each canonical name AND each alias becomes its own pattern entry, so all
    surface forms in a job description are matched regardless of capitalisation.

    Pattern schema (spaCy phrase pattern):
        {
            "label": "SKILL",       # entity label — always SKILL_ENTITY_LABEL
            "pattern": "React.js",  # surface string spaCy will phrase-match
            "id": "t-101"           # taxonomy ID — accessible via ent.ent_id_
        }

    Args:
        taxonomy_entries: List of validated taxonomy entry dicts from loader.

    Returns:
        List of spaCy EntityRuler pattern dicts.
    """
    patterns: list[dict] = []
    seen: set[str] = set()  # track (pattern_lower, id) pairs to avoid duplicates

    for entry in taxonomy_entries:
        canonical = entry["canonical_name"]
        tax_id = entry.get("id", canonical)

        # All surface forms = canonical name + all aliases
        surface_forms = [canonical] + list(entry.get("aliases", []))

        for form in surface_forms:
            form = form.strip()
            if not form:
                continue

            dedup_key = (form.lower(), tax_id)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            patterns.append({
                "label": SKILL_ENTITY_LABEL,
                "pattern": form,
                "id": tax_id,
            })

    logger.debug(
        f"[entity_ruler] Built {len(patterns)} patterns "
        f"from {len(taxonomy_entries)} taxonomy entries."
    )
    return patterns


# ---------------------------------------------------------------------------
# Ruler Attachment
# ---------------------------------------------------------------------------

def attach_entity_ruler(
    nlp: Language,
    taxonomy_entries: list[TaxonomyEntry] | None = None,
) -> Any:
    """
    Builds patterns from the taxonomy and attaches a configured EntityRuler
    to the provided spaCy nlp pipeline object (in-place).

    Configuration:
      - `phrase_matcher_attr="LOWER"`: case-insensitive phrase matching.
      - `overwrite_ents=True`: our skill rules override statistical NER entities.

    Args:
        nlp             : A spaCy Language object (blank or pre-loaded model).
        taxonomy_entries: Optional list of taxonomy entries. Loads from disk
                          automatically if not provided.

    Returns:
        The configured EntityRuler component (already added to nlp.pipe).

    Raises:
        ValueError: If taxonomy_entries is an empty list.
    """
    if taxonomy_entries is None:
        taxonomy_entries = load_taxonomy()

    if not taxonomy_entries:
        raise ValueError(
            "[entity_ruler] Cannot build EntityRuler: taxonomy_entries is empty."
        )

    patterns = build_patterns(taxonomy_entries)

    # spaCy 3.x: add_pipe returns the component and mutates nlp in-place
    ruler = nlp.add_pipe(
        "entity_ruler",
        config={
            "phrase_matcher_attr": "LOWER",  # case-insensitive matching
            "overwrite_ents": True,           # our rules win over statistical NER
        },
        first=True,  # runs before any other component
    )
    ruler.add_patterns(patterns)

    logger.info(
        f"[entity_ruler] EntityRuler attached: "
        f"{len(patterns)} patterns | phrase_matcher_attr=LOWER"
    )
    return ruler
