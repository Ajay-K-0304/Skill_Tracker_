"""
Phase 3 Unit Tests: 6-to-12-Month Time-Series Forecasting Engine.
"""

import pytest
from services.ai_forecasting.data.synthetic_generator import SyntheticDemoDataSource
from services.ai_forecasting.forecasting.engine import SklearnTimeSeriesForecaster
from services.ai_forecasting.schemas import HistoricalObservation


def test_forecast_6_months_sufficient_data():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    python_obs = series["Python"]

    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("Python", python_obs, horizon_months=6)

    assert status == "success"
    assert len(points) == 6
    assert points[0].year_month == "2026-01"
    assert points[-1].year_month == "2026-06"

    for pt in points:
        assert pt.predicted_count >= 0.0
        assert pt.lower_bound is not None and pt.lower_bound >= 0.0
        assert pt.upper_bound is not None and pt.upper_bound >= pt.lower_bound

    # Because N=24 >= 24, 6-month holdout metrics are calculated (18m train, 6m test)
    assert metrics.mae is not None
    assert metrics.rmse is not None
    assert metrics.mae >= 0.0
    assert metrics.rmse >= 0.0
    assert metrics.evaluation_samples == 6
    assert "6 months test" in metrics.notes


def test_forecast_12_months_horizon():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    fastapi_obs = series["FastAPI"]

    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("FastAPI", fastapi_obs, horizon_months=12)

    assert status == "success"
    assert len(points) == 12
    assert points[0].year_month == "2026-01"
    assert points[-1].year_month == "2026-12"


def test_forecast_insufficient_data():
    short_obs = [
        HistoricalObservation(year_month="2026-01", count=10, total_jobs=100),
        HistoricalObservation(year_month="2026-02", count=12, total_jobs=105),
    ]
    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("Python", short_obs, horizon_months=6)

    assert status == "insufficient_data"
    assert len(points) == 0
    assert metrics.mae is None
    assert any("Insufficient historical data" in lim for lim in limitations)


def test_forecast_short_history_withholds_holdout():
    # 6 observations: sufficient for forecast (>= 4), but insufficient for holdout split (< 12)
    short_obs = [
        HistoricalObservation(year_month=f"2026-{m:02d}", count=10 + m, total_jobs=100)
        for m in range(1, 7)
    ]
    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("Python", short_obs, horizon_months=6)

    assert len(points) == 6
    assert metrics.mae is None
    assert metrics.rmse is None
    assert any("Holdout sample too small" in lim for lim in limitations)


def test_forecast_non_negative_clamp_on_declining_series():
    # Steeply declining observations that would extrapolate into negative numbers
    declining_obs = [
        HistoricalObservation(year_month=f"2025-{m:02d}", count=max(1, 80 - m * 15), total_jobs=200)
        for m in range(1, 7)
    ]
    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("LegacyTool", declining_obs, horizon_months=12)

    for pt in points:
        assert pt.predicted_count >= 0.0
        assert pt.lower_bound >= 0.0


def test_forecast_zero_demand_series():
    zero_obs = [
        HistoricalObservation(year_month=f"2025-{m:02d}", count=0, total_jobs=100)
        for m in range(1, 7)
    ]
    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("UnusedSkill", zero_obs, horizon_months=6)

    assert len(points) == 6
    for pt in points:
        assert pt.predicted_count == 0.0
        assert pt.lower_bound == 0.0
        assert pt.upper_bound == 0.0


def test_forecast_current_anchored_synthetic_data_future_dates():
    from datetime import datetime, timezone
    from services.ai_forecasting.data.aggregator import add_months

    source = SyntheticDemoDataSource(months_count=24, seed=42)
    series, _ = source._generate()
    fastapi_obs = series["FastAPI"]

    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    last_hist = fastapi_obs[-1].year_month
    assert last_hist == current_ym

    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("FastAPI", fastapi_obs, horizon_months=6)

    assert status == "success"
    assert len(points) == 6
    # First forecast point MUST be the month immediately following the current month
    assert points[0].year_month == add_months(current_ym, 1)
    assert points[-1].year_month == add_months(current_ym, 6)

    # Every point must be strictly after the historical observation end and current month
    for pt in points:
        assert pt.year_month > last_hist
        assert pt.year_month > current_ym


def test_forecast_12_months_anchored_origin():
    from datetime import datetime, timezone
    from services.ai_forecasting.data.aggregator import add_months

    source = SyntheticDemoDataSource(months_count=24, seed=42)
    series, _ = source._generate()
    py_obs = series["Python"]

    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("Python", py_obs, horizon_months=12)

    assert len(points) == 12
    assert points[0].year_month == add_months(current_ym, 1)
    assert points[-1].year_month == add_months(current_ym, 12)


def test_forecast_adaptive_holdout_tradeoff_12_vs_24_months():
    forecaster = SklearnTimeSeriesForecaster()

    # Case A: 24-month series -> 6 months holdout (18m train, 6m test)
    obs_24 = [
        HistoricalObservation(year_month=f"2024-{(m % 12) + 1:02d}" if m < 12 else f"2025-{(m % 12) + 1:02d}", count=50 + m, total_jobs=200)
        for m in range(24)
    ]
    # fix YYYY-MM
    from services.ai_forecasting.data.aggregator import add_months
    for idx, obs in enumerate(obs_24):
        obs.year_month = add_months("2024-01", idx)

    pts24, met24, _, _ = forecaster.forecast("Tech24", obs_24, horizon_months=6)
    assert met24.evaluation_samples == 6
    assert "6 months test" in met24.notes
    assert "18 months train" in met24.notes

    # Case B: 12-month series -> 3 months holdout (9m train, 3m test)
    obs_12 = obs_24[:12]
    pts12, met12, _, _ = forecaster.forecast("Tech12", obs_12, horizon_months=6)
    assert met12.evaluation_samples == 3
    assert "3 months test" in met12.notes
    assert "9 months train" in met12.notes


def test_forecast_12_months_extrapolation_rising_stable_declining():
    source = SyntheticDemoDataSource(months_count=24, seed=42)
    series, _ = source._generate()
    forecaster = SklearnTimeSeriesForecaster()

    # 1. Rising Series (FastAPI)
    fa_pts, fa_met, _, _ = forecaster.forecast("FastAPI", series["FastAPI"], horizon_months=12)
    assert len(fa_pts) == 12
    # Verify that prediction interval margin widens into the future: margin = upper - pred
    margin_m1 = fa_pts[0].upper_bound - fa_pts[0].predicted_count
    margin_m12 = fa_pts[-1].upper_bound - fa_pts[-1].predicted_count
    assert margin_m12 > margin_m1
    # Check positive trajectory: month 12 predicted count exceeds month 1
    assert fa_pts[-1].predicted_count > fa_pts[0].predicted_count

    # 2. Stable Series (Python)
    py_pts, py_met, _, _ = forecaster.forecast("Python", series["Python"], horizon_months=12)
    assert len(py_pts) == 12
    # Python fluctuates within healthy corridor without collapsing
    for pt in py_pts:
        assert 100.0 <= pt.predicted_count <= 160.0

    # 3. Declining Series (jQuery)
    jq_pts, jq_met, _, _ = forecaster.forecast("jQuery", series["jQuery"], horizon_months=12)
    assert len(jq_pts) == 12
    # Historical demand dropped from ~92 to ~24; 12-month extrapolation reaches 0
    assert jq_pts[-1].predicted_count == 0.0
    for pt in jq_pts:
        assert pt.predicted_count >= 0.0
        assert pt.lower_bound >= 0.0
        assert pt.upper_bound >= pt.lower_bound
        assert pt.upper_bound >= pt.predicted_count


def test_forecast_bounds_and_non_negativity_steep_drop():
    # A series dropping to 0 and extrapolating far into negative territory
    from services.ai_forecasting.data.aggregator import add_months
    steep_obs = [
        HistoricalObservation(year_month=add_months("2025-01", i), count=max(1, 120 - i * 15), total_jobs=300)
        for i in range(12)
    ]
    forecaster = SklearnTimeSeriesForecaster()
    points, metrics, status, limitations = forecaster.forecast("CollapsingSkill", steep_obs, horizon_months=12)

    assert len(points) == 12
    for pt in points:
        assert pt.predicted_count >= 0.0
        assert pt.lower_bound >= 0.0
        assert pt.upper_bound >= pt.predicted_count
        assert pt.upper_bound >= pt.lower_bound

