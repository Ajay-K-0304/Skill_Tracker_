"""
contract_formatter.py — Contract A Output Formatter & Schema Validator
========================================================================
Member 2 | Module: Skill Ontology & NLP Extraction
Branch  : feature/nlp-ontology

Responsibilities:
  1. Assembles extracted canonical skills into the strict Contract A JSON
     schema defined in CONTEXT.md:
        {
          "job_id": "12345",
          "extracted_skills": ["Docker", "FastAPI", "Python"],
          "posting_date": "2026-10-01"
        }
  2. Validates schema compliance:
       - Required keys: "job_id", "extracted_skills", "posting_date"
       - Correct types and constraints (job_id non-empty str, posting_date ISO-8601 YYYY-MM-DD,
         extracted_skills list of unique strings).
  3. Provides batch formatting and JSON file export for downstream consumers:
       - Member 1 (Ingestion / Database storage)
       - Member 3 (Time-Series & Forecasting Engine)

Zero external dependencies outside Python stdlib + internal config / normalizer.
"""

from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional, Union

from services.nlp_engine.config import (
    DATE_FORMAT,
    LOG_LEVEL,
)
from services.nlp_engine.normalization.skill_normalizer import (
    SkillInput,
    to_canonical_list,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Contract A Schema
# ---------------------------------------------------------------------------

CONTRACT_A_REQUIRED_KEYS = ("job_id", "extracted_skills", "posting_date")
DATE_REGEX = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ContractValidationError(ValueError):
    """Raised when a data payload violates Contract A schema constraints."""
    pass


# ---------------------------------------------------------------------------
# Schema Validator
# ---------------------------------------------------------------------------

def validate_contract_a(payload: dict[str, Any], strict: bool = True) -> bool:
    """
    Validates that a dictionary strictly adheres to the Contract A JSON schema.

    Schema Constraints:
      1. Must be a dictionary.
      2. Must contain all required keys: 'job_id', 'extracted_skills', 'posting_date'.
      3. If strict=True, must not contain any unexpected extra top-level keys.
      4. 'job_id' must be a non-empty string.
      5. 'posting_date' must be a valid ISO 8601 calendar date string ('YYYY-MM-DD').
      6. 'extracted_skills' must be a list of non-empty strings.

    Args:
        payload: Dict to validate.
        strict : If True, disallows unknown top-level keys. Default True.

    Returns:
        True if the payload is valid.

    Raises:
        ContractValidationError: If any constraint is violated, with a clear description.
    """
    if not isinstance(payload, dict):
        raise ContractValidationError(
            f"Contract A payload must be a dict, got {type(payload).__name__}."
        )

    # 1. Check required keys
    for key in CONTRACT_A_REQUIRED_KEYS:
        if key not in payload:
            raise ContractValidationError(
                f"Missing required key '{key}' in Contract A payload: {payload}"
            )

    # 2. Strict key checking
    if strict:
        extra_keys = set(payload.keys()) - set(CONTRACT_A_REQUIRED_KEYS)
        if extra_keys:
            raise ContractValidationError(
                f"Unexpected extra keys {sorted(list(extra_keys))} in Contract A payload."
            )

    # 3. Validate 'job_id'
    job_id = payload["job_id"]
    if not isinstance(job_id, str) or not job_id.strip():
        raise ContractValidationError(
            f"'job_id' must be a non-empty string. Got: {job_id!r}"
        )

    # 4. Validate 'posting_date'
    posting_date = payload["posting_date"]
    if not isinstance(posting_date, str) or not DATE_REGEX.match(posting_date):
        raise ContractValidationError(
            f"'posting_date' must be an ISO 8601 string in 'YYYY-MM-DD' format. "
            f"Got: {posting_date!r}"
        )

    # Validate that it is a real calendar date (e.g. not 2026-02-31)
    try:
        datetime.strptime(posting_date, DATE_FORMAT)
    except ValueError as e:
        raise ContractValidationError(
            f"'posting_date' is not a valid calendar date: {posting_date!r} ({e})"
        )

    # 5. Validate 'extracted_skills'
    extracted_skills = payload["extracted_skills"]
    if not isinstance(extracted_skills, list):
        raise ContractValidationError(
            f"'extracted_skills' must be a list. Got: {type(extracted_skills).__name__}"
        )

    for i, skill in enumerate(extracted_skills):
        if not isinstance(skill, str) or not skill.strip():
            raise ContractValidationError(
                f"'extracted_skills[{i}]' must be a non-empty string. Got: {skill!r}"
            )

    return True


# ---------------------------------------------------------------------------
# Date Normalizer Helper
# ---------------------------------------------------------------------------

def _normalize_date(date_val: Optional[Union[str, date, datetime]]) -> str:
    """
    Normalizes dates, datetimes, or date strings into an ISO 'YYYY-MM-DD' string.
    If None, returns today's UTC date.
    """
    if date_val is None:
        return datetime.now(timezone.utc).strftime(DATE_FORMAT)

    if isinstance(date_val, (date, datetime)):
        return date_val.strftime(DATE_FORMAT)

    if isinstance(date_val, str):
        date_str = date_val.strip()
        # Direct match for YYYY-MM-DD
        if DATE_REGEX.match(date_str):
            try:
                datetime.strptime(date_str, DATE_FORMAT)
                return date_str
            except ValueError:
                pass

        # Try common alternative formats
        for fmt in ("%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(date_str[:19], fmt)
                return dt.strftime(DATE_FORMAT)
            except ValueError:
                continue

    raise ContractValidationError(
        f"Unable to parse '{date_val}' as a valid date. Expected format: YYYY-MM-DD."
    )


# ---------------------------------------------------------------------------
# Public Formatter API
# ---------------------------------------------------------------------------

def format_contract_a(
    job_id: Union[str, int],
    extracted_skills: list[str],
    posting_date: Optional[Union[str, date, datetime]] = None,
    sort_skills: bool = True,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Constructs a validated Contract A JSON record for a single job posting.

    Args:
        job_id          : Identifier for the job posting (e.g. "12345" or 12345).
        extracted_skills: List of canonical skill names (e.g. ["Python", "FastAPI"]).
        posting_date    : Date of the posting (string, date, or datetime).
                          Defaults to current UTC date if None.
        sort_skills     : Whether to sort skills alphabetically. Default True.
        validate        : Whether to validate against schema before returning. Default True.

    Returns:
        Contract A compliant dictionary:
            {
                "job_id": "12345",
                "extracted_skills": ["Docker", "FastAPI", "Python"],
                "posting_date": "2026-10-01"
            }

    Raises:
        ContractValidationError: If inputs cannot be formatted into a valid schema.
    """
    # Normalize job_id to string
    if job_id is None:
        raise ContractValidationError("job_id cannot be None.")
    str_job_id = str(job_id).strip()
    if not str_job_id:
        raise ContractValidationError("job_id cannot be empty or whitespace.")

    # Normalize posting date
    normalized_date = _normalize_date(posting_date)

    # Clean and deduplicate skills
    if not isinstance(extracted_skills, list):
        raise ContractValidationError(
            f"extracted_skills must be a list, got {type(extracted_skills).__name__}"
        )

    unique_skills = list(
        dict.fromkeys(
            [str(s).strip() for s in extracted_skills if s is not None and str(s).strip()]
        )
    )
    if sort_skills:
        unique_skills.sort()

    payload = {
        "job_id": str_job_id,
        "extracted_skills": unique_skills,
        "posting_date": normalized_date,
    }

    if validate:
        validate_contract_a(payload)

    return payload


def format_from_raw_skills(
    job_id: Union[str, int],
    raw_skills_or_hits: list[SkillInput],
    posting_date: Optional[Union[str, date, datetime]] = None,
    allow_fuzzy: bool = False,
    validate: bool = True,
) -> dict[str, Any]:
    """
    Convenience end-to-end formatter: takes raw skill mentions or NER hits,
    normalizes them via SkillNormalizer, and outputs a valid Contract A record.

    Args:
        job_id            : Job identifier.
        raw_skills_or_hits: Raw skill strings (e.g. ["React.js", "python3"]) or SkillHit dicts.
        posting_date      : Posting date.
        allow_fuzzy       : Whether to allow fuzzy matching in normalizer.
        validate          : Whether to validate output.

    Returns:
        Contract A compliant dict.
    """
    canonical_skills = to_canonical_list(
        raw_skills_or_hits,
        deduplicate=True,
        sort_result=True,
        allow_fuzzy=allow_fuzzy,
    )
    return format_contract_a(
        job_id=job_id,
        extracted_skills=canonical_skills,
        posting_date=posting_date,
        sort_skills=False,  # already sorted by to_canonical_list
        validate=validate,
    )


def format_contract_a_batch(
    records: list[dict[str, Any]],
    validate: bool = True,
) -> list[dict[str, Any]]:
    """
    Formats a batch of job records into Contract A records.

    Expected input per item:
      {"job_id": ..., "extracted_skills": [...], "posting_date": ...}
      (or "skills" in place of "extracted_skills")

    Args:
        records : List of job dictionaries.
        validate: Whether to validate each record.

    Returns:
        List of Contract A formatted dictionaries.
    """
    if not records:
        return []

    results: list[dict[str, Any]] = []
    for item in records:
        job_id = item.get("job_id")
        skills = item.get("extracted_skills", item.get("skills", []))
        p_date = item.get("posting_date", item.get("date"))

        formatted = format_contract_a(
            job_id=job_id,
            extracted_skills=skills,
            posting_date=p_date,
            validate=validate,
        )
        results.append(formatted)

    return results


# ---------------------------------------------------------------------------
# JSON Export Helpers
# ---------------------------------------------------------------------------

def to_contract_json(
    data: Union[dict[str, Any], list[dict[str, Any]]],
    indent: int = 2,
) -> str:
    """Serializes Contract A record(s) to a formatted JSON string."""
    return json.dumps(data, indent=indent, ensure_ascii=False)


def save_contract_json(
    data: Union[dict[str, Any], list[dict[str, Any]]],
    filepath: Union[str, Path],
    indent: int = 2,
) -> Path:
    """
    Saves Contract A record(s) to a JSON file on disk.

    Args:
        data    : Single Contract A dict or list of dicts.
        filepath: Destination file path.
        indent  : JSON indent. Default 2.

    Returns:
        Resolved Path of saved file.
    """
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    logger.info(f"[contract_formatter] Saved Contract A JSON to {path}")
    return path
