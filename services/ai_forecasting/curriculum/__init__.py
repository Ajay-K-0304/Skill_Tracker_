"""
Curriculum gap analysis and role benchmark models.
"""

from services.ai_forecasting.curriculum.benchmarks import get_target_role_benchmark, ROLE_BENCHMARKS
from services.ai_forecasting.curriculum.gap_analyzer import analyze_skill_gap

__all__ = ["get_target_role_benchmark", "ROLE_BENCHMARKS", "analyze_skill_gap"]
