"""
services/nlp_engine/normalization/__init__.py
=============================================
Exposes the skill normalizer engine, normalized types, and convenience functions.
"""

from services.nlp_engine.normalization.skill_normalizer import (
    NormalizedSkill,
    SkillInput,
    SkillNormalizer,
    get_default_normalizer,
    normalize,
    normalize_list,
    to_canonical_list,
)

__all__ = [
    "NormalizedSkill",
    "SkillInput",
    "SkillNormalizer",
    "get_default_normalizer",
    "normalize",
    "normalize_list",
    "to_canonical_list",
]
