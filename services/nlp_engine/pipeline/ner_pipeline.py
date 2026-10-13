"""
ner_pipeline.py — Main NER Runner
===================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  Orchestrates the full NER flow for a single job description or a batch:
    cleaned text  →  spaCy Doc  →  raw skill hit list

  Returns RAW surface-form skill hits (e.g. "React.js", "python3").
  Normalization to canonical names ("React", "Python") is Phase 4's job.

Output schema per skill hit:
    {
        "text"       : "React.js",   # surface form found in text
        "label"      : "SKILL",      # always SKILL_ENTITY_LABEL
        "start_char" : 12,           # character offset (start) in cleaned text
        "end_char"   : 20,           # character offset (end) in cleaned text
        "taxonomy_id": "t-101"       # links to skill_taxonomy entry id
    }

Decoupling / testability:
  NERPipeline accepts an optional pre-built `spacy.Language` object so tests
  can inject `spacy.blank("en")` without needing a downloaded model.
  Production code calls `NERPipeline()` with no args and loads SPACY_MODEL.

This module has ZERO imports outside stdlib + spaCy + config/pipeline modules.
"""

import logging
from typing import Optional

import spacy
from spacy.language import Language

from services.nlp_engine.config import (
    LOG_LEVEL,
    SKILL_ENTITY_LABEL,
    SPACY_DISABLED_COMPONENTS,
    SPACY_MODEL,
)
from services.nlp_engine.pipeline.entity_ruler import attach_entity_ruler
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
# Type alias for a single skill hit
# ---------------------------------------------------------------------------

SkillHit = dict  # keys: text, label, start_char, end_char, taxonomy_id


# ---------------------------------------------------------------------------
# Internal: spaCy model loader
# ---------------------------------------------------------------------------

def _load_spacy_model(model_name: str) -> Language:
    """
    Loads a spaCy model by name with non-essential components disabled.
    Falls back to a blank English model if the named model is not installed,
    logging a clear warning so the user knows to run:
        python -m spacy download en_core_web_sm

    Args:
        model_name: spaCy model string (e.g. "en_core_web_sm").

    Returns:
        A spaCy Language object ready for EntityRuler attachment.
    """
    try:
        nlp = spacy.load(model_name, disable=SPACY_DISABLED_COMPONENTS)
        logger.info(f"[ner_pipeline] Loaded spaCy model: '{model_name}'")
        return nlp
    except OSError:
        logger.warning(
            f"[ner_pipeline] spaCy model '{model_name}' not found. "
            "Falling back to spacy.blank('en'). "
            f"Run: python -m spacy download {model_name}"
        )
        return spacy.blank("en")


# ---------------------------------------------------------------------------
# NERPipeline Class
# ---------------------------------------------------------------------------

class NERPipeline:
    """
    Stateful NER pipeline that wraps a spaCy model + EntityRuler.

    Instantiate once (model load is expensive) and call `extract()` or
    `extract_batch()` repeatedly.

    Attributes:
        _nlp: The configured spaCy Language pipeline.

    Usage (production):
        pipeline = NERPipeline()
        hits = pipeline.extract("We need Python and React.js experience.")

    Usage (testing — no model download needed):
        import spacy
        nlp = spacy.blank("en")
        pipeline = NERPipeline(nlp=nlp)
        hits = pipeline.extract("We need Python and React.js experience.")
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        taxonomy_entries: Optional[list[TaxonomyEntry]] = None,
    ) -> None:
        """
        Initializes the NER pipeline.

        Args:
            nlp             : Optional pre-built spaCy Language object.
                              If None, loads SPACY_MODEL from config.
                              Inject `spacy.blank("en")` for tests.
            taxonomy_entries: Optional taxonomy list. Loads from disk if None.
        """
        if nlp is None:
            self._nlp = _load_spacy_model(SPACY_MODEL)
        else:
            self._nlp = nlp

        # Attach EntityRuler with all taxonomy patterns
        entries = taxonomy_entries if taxonomy_entries is not None else load_taxonomy()
        attach_entity_ruler(self._nlp, taxonomy_entries=entries)

        logger.info(
            f"[ner_pipeline] NERPipeline ready. "
            f"Pipe stages: {self._nlp.pipe_names}"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _doc_to_hits(doc) -> list[SkillHit]:
        """
        Extracts all SKILL entities from a spaCy Doc and returns them
        as a list of SkillHit dicts.

        Deduplicates hits by (text.lower(), taxonomy_id) so the same skill
        mentioned multiple times in one job description is returned once.

        Args:
            doc: A processed spaCy Doc object.

        Returns:
            Deduplicated list of SkillHit dicts for all SKILL entities found.
        """
        hits: list[SkillHit] = []
        seen: set[str] = set()

        for ent in doc.ents:
            if ent.label_ != SKILL_ENTITY_LABEL:
                continue

            # Deduplicate by taxonomy_id only — so "Python" and "python3"
            # (same taxonomy entry t-001) produce a single hit per doc.
            dedup_key = ent.ent_id_
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            hits.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "taxonomy_id": ent.ent_id_,
            })

        return hits

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def extract(self, text: str) -> list[SkillHit]:
        """
        Runs NER on a single cleaned job description and returns skill hits.

        Args:
            text: Clean plain-text job description (output of text_cleaner.clean()).
                  Must be a non-empty string.

        Returns:
            List of SkillHit dicts (may be empty if no skills detected).
            Each hit: {text, label, start_char, end_char, taxonomy_id}

        Example:
            >>> pipeline.extract("We need React.js and Python developers.")
            [
                {"text": "React.js", "label": "SKILL", "start_char": 8,
                 "end_char": 16, "taxonomy_id": "t-101"},
                {"text": "Python", "label": "SKILL", "start_char": 21,
                 "end_char": 27, "taxonomy_id": "t-001"},
            ]
        """
        if not text or not text.strip():
            logger.debug("[ner_pipeline] Empty text received, returning no hits.")
            return []

        doc = self._nlp(text)
        hits = self._doc_to_hits(doc)

        logger.debug(
            f"[ner_pipeline] Extracted {len(hits)} skill(s) from text "
            f"({len(text)} chars)."
        )
        return hits

    def extract_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[SkillHit]]:
        """
        Runs NER on a list of cleaned job descriptions using spaCy's
        efficient `nlp.pipe()` for batch processing.

        Args:
            texts     : List of cleaned job description strings.
            batch_size: spaCy batch size for pipe(). Default 64.

        Returns:
            List of SkillHit lists, one per input text (preserves order).
            Empty texts produce empty hit lists.

        Example:
            >>> pipeline.extract_batch(["Python dev", "React.js needed"])
            [
                [{"text": "Python", ...}],
                [{"text": "React.js", ...}]
            ]
        """
        if not texts:
            return []

        results: list[list[SkillHit]] = []

        # Replace empty/None strings with a placeholder so pipe() stays aligned
        safe_texts = [t if (t and t.strip()) else " " for t in texts]

        for doc in self._nlp.pipe(safe_texts, batch_size=batch_size):
            results.append(self._doc_to_hits(doc))

        logger.info(
            f"[ner_pipeline] Batch processed {len(texts)} texts → "
            f"{sum(len(r) for r in results)} total skill hits."
        )
        return results

    @property
    def pipe_names(self) -> list[str]:
        """Returns the list of active component names in the spaCy pipeline."""
        return self._nlp.pipe_names
