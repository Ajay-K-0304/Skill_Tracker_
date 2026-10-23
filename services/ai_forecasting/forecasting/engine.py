"""
Defensible Time-Series Forecasting Engine using Scikit-Learn.
Features Ridge regression with trend and seasonal harmonics,
widening prediction intervals, non-negative projection enforcement,
and time-ordered holdout evaluation for MAE/RMSE.
"""

from typing import List, Optional, Tuple
import numpy as np
from sklearn.linear_model import Ridge

from services.ai_forecasting.config import settings
from services.ai_forecasting.data.aggregator import add_months
from services.ai_forecasting.forecasting.interfaces import ForecastingEngine
from services.ai_forecasting.schemas import (
    EvaluationMetrics,
    ForecastPoint,
    HistoricalObservation,
)


class SklearnTimeSeriesForecaster(ForecastingEngine):
    """
    Time-series forecasting implementation based on linear/ridge regression
    with polynomial trend, cyclical calendar features, and dynamic uncertainty bands.
    """

    def __init__(
        self,
        min_observations: int = settings.MIN_OBSERVATIONS_FOR_FORECAST,
        holdout_min_observations: int = settings.MIN_OBSERVATIONS_FOR_HOLDOUT,
        alpha: float = 1.0,
    ):
        self.min_observations = min_observations
        self.holdout_min_observations = holdout_min_observations
        self.alpha = alpha

    def _build_features(self, t_indices: np.ndarray, total_points: int) -> np.ndarray:
        """
        Builds feature matrix:
        1. Linear trend t / total_points
        2. Quadratic trend (t / total_points) ** 2
        3. If history >= 12, annual sinusoidal cycles: sin(2pi*t/12) and cos(2pi*t/12)
        """
        norm_t = (t_indices.astype(float)) / max(1, total_points)
        feats = [norm_t, norm_t ** 2]

        if total_points >= 12:
            sin_12 = np.sin(2 * np.pi * t_indices / 12.0)
            cos_12 = np.cos(2 * np.pi * t_indices / 12.0)
            feats.extend([sin_12, cos_12])

        return np.column_stack(feats)

    def _evaluate_holdout(
        self, y: np.ndarray, holdout_size: int = 3
    ) -> EvaluationMetrics:
        """
        Performs time-ordered holdout evaluation without random splitting.
        Trains on y[:-holdout_size] and tests on y[-holdout_size:].
        """
        n = len(y)
        train_y = y[:-holdout_size]
        test_y = y[-holdout_size:]

        train_t = np.arange(len(train_y))
        test_t = np.arange(len(train_y), n)

        X_train = self._build_features(train_t, n)
        X_test = self._build_features(test_t, n)

        model = Ridge(alpha=self.alpha)
        model.fit(X_train, train_y)
        test_pred = np.maximum(0.0, model.predict(X_test))

        mae = float(np.mean(np.abs(test_y - test_pred)))
        rmse = float(np.sqrt(np.mean((test_y - test_pred) ** 2)))

        return EvaluationMetrics(
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            evaluation_samples=holdout_size,
            notes=(
                f"Time-ordered holdout evaluation ({holdout_size} months test; {len(train_y)} months train). "
                f"No random splitting or data leakage."
            ),
        )

    def forecast(
        self,
        skill_name: str,
        observations: List[HistoricalObservation],
        horizon_months: int = 6,
    ) -> Tuple[List[ForecastPoint], EvaluationMetrics, str, List[str]]:
        limitations: List[str] = []
        n_obs = len(observations)

        # 1. Check data sufficiency
        if n_obs < self.min_observations:
            limitations.append(
                f"Insufficient historical data for '{skill_name}': {n_obs} observations available. "
                f"Minimum {self.min_observations} required to construct a valid forecast."
            )
            empty_metrics = EvaluationMetrics(
                notes=f"Evaluation skipped: only {n_obs} observations available."
            )
            return [], empty_metrics, "insufficient_data", limitations

        # 2. Extract values and dates
        sorted_obs = sorted(observations, key=lambda x: x.year_month)
        last_ym = sorted_obs[-1].year_month
        y = np.array([float(obs.count) for obs in sorted_obs], dtype=float)

        # 3. Handle constant or zero series
        if np.all(y == 0):
            limitations.append(f"Skill '{skill_name}' has 0 recorded demand across all historical months.")
            zero_forecast = [
                ForecastPoint(
                    year_month=add_months(last_ym, step),
                    predicted_count=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                )
                for step in range(1, horizon_months + 1)
            ]
            zero_metrics = EvaluationMetrics(
                mae=0.0,
                rmse=0.0,
                evaluation_samples=n_obs,
                notes="Zero demand series: predictions clamped to 0.",
            )
            return zero_forecast, zero_metrics, "warning", limitations

        # 4. Holdout evaluation (only when history is sufficient)
        if n_obs >= self.holdout_min_observations:
            # For 24+ months, evaluate a 6-month holdout (25% split: 18m train, 6m test).
            # For shorter series (12-23m), reserve at least 75% for training (evaluating 3m).
            holdout_k = 6 if n_obs >= 24 else min(3, max(1, n_obs // 4))
            metrics = self._evaluate_holdout(y, holdout_size=holdout_k)
        else:
            limitations.append(
                f"Holdout sample too small ({n_obs} months). Meaningful MAE/RMSE calculation withheld."
            )
            metrics = EvaluationMetrics(
                mae=None,
                rmse=None,
                evaluation_samples=None,
                notes=f"Sample size ({n_obs} months) < threshold ({self.holdout_min_observations}) for holdout evaluation.",
            )

        # 5. Fit full model on all available history
        t_hist = np.arange(n_obs)
        X_hist = self._build_features(t_hist, n_obs)

        model = Ridge(alpha=self.alpha)
        model.fit(X_hist, y)
        fitted_values = model.predict(X_hist)

        # 6. Estimate residual standard deviation for prediction intervals
        residuals = y - fitted_values
        dof = max(1, n_obs - X_hist.shape[1])
        residual_std = float(np.sqrt(np.sum(residuals ** 2) / dof))
        residual_std = max(1.0, residual_std)  # floor at 1.0 posting for realistic band

        # 7. Generate future points
        t_future = np.arange(n_obs, n_obs + horizon_months)
        X_future = self._build_features(t_future, n_obs)
        raw_predictions = model.predict(X_future)

        forecast_points: List[ForecastPoint] = []
        for step_idx, pred in enumerate(raw_predictions, start=1):
            future_ym = add_months(last_ym, step_idx)

            # Prediction interval widens into the future: sqrt(1 + h/N)
            widening_factor = np.sqrt(1.0 + (step_idx / float(n_obs)))
            margin = 1.96 * residual_std * widening_factor

            # Strictly enforce non-negative counts and bounds
            predicted_count = max(0.0, round(float(pred), 1))
            lower_bound = max(0.0, round(float(pred - margin), 1))
            upper_bound = max(predicted_count, round(float(pred + margin), 1))

            forecast_points.append(
                ForecastPoint(
                    year_month=future_ym,
                    predicted_count=predicted_count,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            )

        status = "success" if not limitations else "warning"
        return forecast_points, metrics, status, limitations
