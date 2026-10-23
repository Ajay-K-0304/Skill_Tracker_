"""
Abstract interface for forecasting engines.
Enables pluggable model architectures (Scikit-Learn, Prophet, Statsmodels).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from services.ai_forecasting.schemas import (
    EvaluationMetrics,
    ForecastPoint,
    HistoricalObservation,
)


class ForecastingEngine(ABC):
    """Protocol for a skill demand forecasting model."""

    @abstractmethod
    def forecast(
        self,
        skill_name: str,
        observations: List[HistoricalObservation],
        horizon_months: int = 6,
    ) -> Tuple[List[ForecastPoint], EvaluationMetrics, str, List[str]]:
        """
        Generates predictions for a single skill.
        Returns:
            Tuple of:
            - forecast_points: List of ForecastPoint (future year_month, predicted_count, lower_bound, upper_bound)
            - metrics: EvaluationMetrics (holdout MAE/RMSE if data >= 12 months, else sample limitation notes)
            - status: "success" | "insufficient_data" | "warning"
            - limitations: List of limitations/warnings
        """
        pass
