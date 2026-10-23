"""
Deterministic Skill Trajectory Classifier.
Classifies skills into 'Emerging', 'Core/Stable', 'Declining', 'Niche', or 'Insufficient Data'
using transparent mathematical thresholds and historical time-series evidence.
"""

from typing import List, Optional
import numpy as np

from services.ai_forecasting.config import settings
from services.ai_forecasting.schemas import (
    ForecastPoint,
    HistoricalObservation,
    TrajectoryLabelType,
    TrajectoryResult,
)


class TrajectoryClassifier:
    """Classifies skill demand trajectories based on configurable statistical thresholds."""

    def __init__(
        self,
        min_observations: int = settings.MIN_OBSERVATIONS_FOR_FORECAST,
        emerging_threshold: float = settings.EMERGING_GROWTH_THRESHOLD,
        declining_threshold: float = settings.DECLINING_GROWTH_THRESHOLD,
        niche_max_share: float = settings.NICHE_MAX_SHARE_THRESHOLD,
        min_volume_for_core: int = settings.MIN_VOLUME_FOR_CORE,
    ):
        self.min_observations = min_observations
        self.emerging_threshold = emerging_threshold
        self.declining_threshold = declining_threshold
        self.niche_max_share = niche_max_share
        self.min_volume_for_core = min_volume_for_core

    def classify(
        self,
        skill_name: str,
        observations: List[HistoricalObservation],
        forecast_points: Optional[List[ForecastPoint]] = None,
    ) -> TrajectoryResult:
        n = len(observations)

        # 1. Check data sufficiency
        if n < self.min_observations:
            return TrajectoryResult(
                label="Insufficient Data",
                growth_rate=None,
                recent_vs_earlier_ratio=None,
                summary=(
                    f"Insufficient historical data ({n} months). "
                    f"Minimum {self.min_observations} months required to establish trajectory."
                ),
                data_sufficiency=False,
            )

        sorted_obs = sorted(observations, key=lambda x: x.year_month)
        counts = np.array([float(o.count) for o in sorted_obs])
        shares = [o.demand_share for o in sorted_obs if o.demand_share is not None]
        avg_share = float(np.mean(shares)) if shares else 0.0
        overall_mean = float(np.mean(counts))

        # 2. Earlier vs Recent demand comparison
        half = max(1, n // 2)
        earlier_counts = counts[:half]
        recent_counts = counts[half:]

        mean_earlier = float(np.mean(earlier_counts))
        mean_recent = float(np.mean(recent_counts))

        if mean_earlier > 0:
            growth_rate = round((mean_recent - mean_earlier) / mean_earlier, 3)
            ratio = round(mean_recent / mean_earlier, 3)
        else:
            growth_rate = 1.0 if mean_recent > 0 else 0.0
            ratio = 2.0 if mean_recent > 0 else 1.0

        min_count = int(np.min(counts))
        max_count = int(np.max(counts))
        std_count = float(np.std(counts))

        # 3. Classify based on thresholds
        # A. Niche: consistently low overall demand volume and low market share
        if overall_mean < self.min_volume_for_core and avg_share <= self.niche_max_share:
            label: TrajectoryLabelType = "Niche"
            summary = (
                f"Niche skill: low overall monthly demand averaging {overall_mean:.1f} postings/month "
                f"(range: {min_count}-{max_count}, avg share {avg_share * 100:.1f}% <= {self.niche_max_share * 100:.0f}% threshold). "
                f"Volume remains below the {self.min_volume_for_core} postings/month core threshold."
            )

        # B. Emerging: significant positive growth and rising market interest
        elif growth_rate >= self.emerging_threshold and mean_recent >= self.min_volume_for_core:
            label = "Emerging"
            summary = (
                f"Emerging skill: strong upward growth of {growth_rate * 100:+.1f}% "
                f"(threshold >= +{self.emerging_threshold * 100:.0f}%, ratio {ratio:.2f}), "
                f"with recent {n - half}-month demand averaging {mean_recent:.1f} postings/month "
                f"(range: {min_count}-{max_count}, overall mean {overall_mean:.1f})."
            )

        # C. Declining: sustained downward contraction
        elif growth_rate <= self.declining_threshold:
            label = "Declining"
            summary = (
                f"Declining skill: downward demand contraction of {growth_rate * 100:.1f}% "
                f"(threshold <= {self.declining_threshold * 100:+.0f}%, ratio {ratio:.2f}), "
                f"from earlier {half}-month avg {mean_earlier:.1f} to recent {n - half}-month avg {mean_recent:.1f} "
                f"(range: {min_count}-{max_count})."
            )

        # D. Core/Stable: robust, persistent demand with manageable fluctuation
        else:
            label = "Core/Stable"
            summary = (
                f"Core/Stable skill: sustained high market presence averaging {overall_mean:.1f} postings/month "
                f"(range: {min_count}-{max_count}, std={std_count:.1f}). "
                f"Comparing recent {n - half}-month average ({mean_recent:.1f}) to earlier {half}-month average ({mean_earlier:.1f}) "
                f"shows stable growth of {growth_rate * 100:+.1f}% (ratio {ratio:.2f}), "
                f"within the [{self.declining_threshold * 100:+.0f}%, +{self.emerging_threshold * 100:.0f}%] stability corridor."
            )

        return TrajectoryResult(
            label=label,
            growth_rate=growth_rate,
            recent_vs_earlier_ratio=ratio,
            summary=summary,
            data_sufficiency=True,
        )


def classify_trajectory(
    skill_name: str,
    observations: List[HistoricalObservation],
    forecast_points: Optional[List[ForecastPoint]] = None,
) -> TrajectoryResult:
    """Convenience function using default configuration."""
    classifier = TrajectoryClassifier()
    return classifier.classify(skill_name, observations, forecast_points)
