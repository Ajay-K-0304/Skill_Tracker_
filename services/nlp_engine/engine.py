"""
engine.py — Unified NLP & Ontology Engine Facade
=================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  High-level orchestrator and entry point uniting the entire NLP pipeline:
    Raw Job Postings
          │
          ▼  [pipeline/text_cleaner.py]
    Clean Plain Text
          │
          ▼  [pipeline/ner_pipeline.py]
    Raw Skill Mentions (SkillHits)
          │
          ▼  [normalization/skill_normalizer.py]
    Canonical Skill Names
          │
     ┌────┴────────────────────────┐
     ▼                             ▼
[output/contract_formatter.py]  [cooccurrence/cooccurrence_matrix.py]
     │                             │
     ▼                             ▼
Contract A Output JSON         Symmetric Co-occurrence Matrix
(for Member 1 & 3)             (for Member 3 & 4)

Usage:
    from services.nlp_engine.engine import NLPEngine

    engine = NLPEngine()

    # Single job posting:
    record = engine.process_job(
        job_id="12345",
        raw_text="<p>Looking for Python and FastAPI developers.</p>",
        posting_date="2026-10-01"
    )

    # Batch processing with co-occurrence computation:
    records, matrix = engine.process_batch(job_list)
"""

from datetime import date, datetime
import logging
from typing import Any, Optional, Union

from spacy.language import Language

from services.nlp_engine.config import (
    LOG_LEVEL,
    MIN_COOCCURRENCE_COUNT,
)
from services.nlp_engine.cooccurrence.cooccurrence_matrix import (
    CooccurrenceMatrix,
)
from services.nlp_engine.normalization.skill_normalizer import (
    SkillNormalizer,
)
from services.nlp_engine.output.contract_formatter import (
    format_contract_a,
    validate_contract_a,
)
from services.nlp_engine.pipeline.ner_pipeline import (
    NERPipeline,
)
from services.nlp_engine.pipeline.text_cleaner import (
    clean,
    clean_batch,
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
# NLPEngine Class
# ---------------------------------------------------------------------------

class NLPEngine:
    """
    Main entry point for Member 2's NLP Engine.

    Encapsulates all processing stages into a simple, production-ready interface.
    Supports dependency injection for offline/model-free unit and integration testing.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        taxonomy_entries: Optional[list[TaxonomyEntry]] = None,
        fuzzy_threshold: float = 0.85,
        min_cooccurrence: int = MIN_COOCCURRENCE_COUNT,
    ) -> None:
        """
        Initializes the NLP engine and all its sub-components.

        Args:
            nlp             : Optional pre-built spaCy pipeline (inject blank model for tests).
            taxonomy_entries: Optional taxonomy list (inject mocks for tests).
            fuzzy_threshold : Similarity threshold for fallback fuzzy matching.
            min_cooccurrence: Minimum count threshold for co-occurrence matrix pairs.
        """
        self.taxonomy = taxonomy_entries if taxonomy_entries is not None else load_taxonomy()

        # Initialize sub-components
        self.ner = NERPipeline(nlp=nlp, taxonomy_entries=self.taxonomy)
        self.normalizer = SkillNormalizer(
            taxonomy_entries=self.taxonomy,
            fuzzy_threshold=fuzzy_threshold,
        )
        self.min_cooccurrence = min_cooccurrence

        logger.info(
            f"[NLPEngine] Initialized with {len(self.taxonomy)} taxonomy skills. "
            f"Active spaCy stages: {self.ner.pipe_names}"
        )

    # -----------------------------------------------------------------------
    # Single Posting Processing
    # -----------------------------------------------------------------------

    def extract_from_text(
        self,
        raw_text: str,
        allow_fuzzy: bool = False,
    ) -> list[str]:
        """
        Extracts canonical skill names from a single raw text description.

        Steps:
          1. Clean text (strip HTML, emails, noise)
          2. Run NER pipeline to detect skill hits
          3. Normalize hits to canonical skill names (deduplicated & sorted)

        Args:
            raw_text   : Raw job description text (can be HTML).
            allow_fuzzy: Whether to allow fuzzy matching fallback.

        Returns:
            Sorted list of canonical skill names: ["Docker", "FastAPI", "Python"].
        """
        if not raw_text or not str(raw_text).strip():
            return []

        # 1. Clean
        cleaned = clean(raw_text)

        # 2. NER extraction
        raw_hits = self.ner.extract(cleaned)

        # 3. Normalization to canonical names
        canonical_skills = self.normalizer.to_canonical_list(
            raw_hits,
            deduplicate=True,
            sort_result=True,
            allow_fuzzy=allow_fuzzy,
        )
        return canonical_skills

    def process_job(
        self,
        job_id: Union[str, int],
        raw_text: str,
        posting_date: Optional[Union[str, date, datetime]] = None,
        allow_fuzzy: bool = False,
    ) -> dict[str, Any]:
        """
        Processes a single job description and returns a validated Contract A JSON record.

        Args:
            job_id      : Unique job identifier.
            raw_text    : Raw job description.
            posting_date: Date of job posting.
            allow_fuzzy : Whether to allow fuzzy matching.

        Returns:
            Contract A compliant dict:
                {
                    "job_id": "12345",
                    "extracted_skills": ["Docker", "FastAPI", "Python"],
                    "posting_date": "2026-10-01"
                }
        """
        canonical_skills = self.extract_from_text(raw_text, allow_fuzzy=allow_fuzzy)
        record = format_contract_a(
            job_id=job_id,
            extracted_skills=canonical_skills,
            posting_date=posting_date,
            sort_skills=False,  # already sorted
            validate=True,
        )
        return record

    # -----------------------------------------------------------------------
    # Batch Processing
    # -----------------------------------------------------------------------

    def process_batch(
        self,
        jobs: list[dict[str, Any]],
        batch_size: int = 64,
        build_cooccurrence: bool = True,
        allow_fuzzy: bool = False,
    ) -> tuple[list[dict[str, Any]], Optional[CooccurrenceMatrix]]:
        """
        High-throughput batch processing for a collection of job postings.
        Utilizes spaCy's optimized nlp.pipe() for multi-threaded processing.

        Args:
            jobs              : List of job dicts. Each dict must contain:
                                - "job_id": ID
                                - "raw_description" or "description": raw text
                                - "posting_date" or "date" (optional): date
            batch_size        : Batch size for nlp.pipe().
            build_cooccurrence: Whether to compute co-occurrence matrix for the batch.
            allow_fuzzy       : Whether to allow fuzzy matching in normalizer.

        Returns:
            Tuple of:
              - List of validated Contract A dictionaries.
              - CooccurrenceMatrix instance (or None if build_cooccurrence=False).
        """
        if not jobs:
            return [], (CooccurrenceMatrix(min_count=self.min_cooccurrence) if build_cooccurrence else None)

        # 1. Extract raw texts and metadata
        raw_texts: list[str] = []
        job_ids: list[Union[str, int]] = []
        dates: list[Any] = []

        for item in jobs:
            text = item.get("raw_description") or item.get("description") or ""
            raw_texts.append(text)
            job_ids.append(item.get("job_id", "unknown"))
            dates.append(item.get("posting_date") or item.get("date"))

        # 2. Batch clean
        cleaned_texts = clean_batch(raw_texts)

        # 3. Batch NER using spaCy nlp.pipe()
        batch_hits = self.ner.extract_batch(cleaned_texts, batch_size=batch_size)

        # 4. Batch normalization
        batch_canonical = self.normalizer.batch_to_canonical_lists(
            batch_hits,
            deduplicate=True,
            sort_result=True,
            allow_fuzzy=allow_fuzzy,
        )

        # 5. Format Contract A records
        contract_records: list[dict[str, Any]] = []
        for j_id, skills, p_date in zip(job_ids, batch_canonical, dates):
            rec = format_contract_a(
                job_id=j_id,
                extracted_skills=skills,
                posting_date=p_date,
                sort_skills=False,
                validate=True,
            )
            contract_records.append(rec)

        # 6. Build co-occurrence matrix if requested
        matrix_model: Optional[CooccurrenceMatrix] = None
        if build_cooccurrence:
            matrix_model = CooccurrenceMatrix(min_count=self.min_cooccurrence)
            matrix_model.fit(batch_canonical)

        logger.info(
            f"[NLPEngine] Batch processing complete: {len(jobs)} jobs processed. "
            f"Total skill mentions extracted across batch: "
            f"{sum(len(r['extracted_skills']) for r in contract_records)}"
        )
        return contract_records, matrix_model
