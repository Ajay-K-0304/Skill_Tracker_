"""
services/nlp_engine/pipeline/__init__.py
========================================
Exposes text cleaning, entity ruler pattern construction, and the NER pipeline.
"""

from services.nlp_engine.pipeline.entity_ruler import (
    attach_entity_ruler,
    build_patterns,
)
from services.nlp_engine.pipeline.ner_pipeline import (
    NERPipeline,
    SkillHit,
)
from services.nlp_engine.pipeline.text_cleaner import (
    clean,
    clean_batch,
)

__all__ = [
    "clean",
    "clean_batch",
    "build_patterns",
    "attach_entity_ruler",
    "NERPipeline",
    "SkillHit",
]
