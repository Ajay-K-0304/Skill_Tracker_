"""
Phase 2 Unit Tests: Monthly skill-demand data aggregation, validation, and adapters.
"""

import asyncio
from pathlib import Path
import pytest
from services.ai_forecasting.data.aggregator import (
    generate_month_range,
    detect_missing_months,
    fill_missing_months,
    validate_monthly_observations,
)
from services.ai_forecasting.data.synthetic_generator import SyntheticDemoDataSource
from services.ai_forecasting.data.csv_loader import CSVDataSource
from services.ai_forecasting.data.db_adapter import LiveDatabaseAdapter
from services.ai_forecasting.schemas import HistoricalObservation


def test_generate_month_range():
    months = generate_month_range("2025-10", "2026-02")
    assert months == ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02"]


def test_detect_missing_months():
    present = ["2025-01", "2025-02", "2025-04", "2025-06"]
    missing = detect_missing_months(present)
    assert missing == ["2025-03", "2025-05"]


def test_fill_missing_months():
    obs = [
        HistoricalObservation(year_month="2025-01", count=10, total_jobs=100),
        HistoricalObservation(year_month="2025-03", count=14, total_jobs=110),
    ]
    totals = {"2025-01": 100, "2025-02": 105, "2025-03": 110}
    filled = fill_missing_months(obs, total_jobs_by_month=totals, zero_fill=True)

    assert len(filled) == 3
    assert filled[0].year_month == "2025-01"
    assert filled[0].count == 10
    assert filled[1].year_month == "2025-02"
    assert filled[1].count == 0  # Zero-filled!
    assert filled[1].total_jobs == 105
    assert filled[2].year_month == "2025-03"
    assert filled[2].count == 14


def test_validate_monthly_observations():
    obs = [
        HistoricalObservation(year_month="2025-01", count=10, total_jobs=100),
        HistoricalObservation(year_month="2025-01", count=10, total_jobs=100),  # Duplicate
    ]
    warnings = validate_monthly_observations(obs)
    assert any("Duplicate" in w for w in warnings)
    assert any("Insufficient history" in w for w in warnings)


def test_synthetic_demo_data_source():
    async def _run():
        source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
        assert source.get_source_label() == "synthetic_demo"

        demand = await source.get_monthly_skill_demand()
        assert "Python" in demand
        assert "FastAPI" in demand
        assert "jQuery" in demand
        assert "Rust" in demand

        # Verify 24 months of data
        assert len(demand["Python"]) == 24
        assert len(demand["FastAPI"]) == 24

        # Verify demand share
        for obs in demand["Python"]:
            assert obs.demand_share is not None
            assert 0.0 <= obs.demand_share <= 1.0

    asyncio.run(_run())


def test_csv_data_source_relational():
    async def _run():
        fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
        source = CSVDataSource(fixtures_dir=fixtures_dir)
        assert source.get_source_label() == "csv"

        demand = await source.get_monthly_skill_demand()
        report = source.get_quality_report()

        # 301 postings, 4 taxonomy skills
        assert report["postings_count"] == 301
        assert report["skills_count"] == 4
        # Warnings about insufficient history (< 4 months) and unlinked postings
        assert any("Historical timespan is too short" in w for w in report["warnings"])
        assert any("have no skills linked" in w for w in report["warnings"])
        assert "Python" in demand

    asyncio.run(_run())


def test_csv_data_source_pre_aggregated():
    async def _run():
        fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"
        pre_agg_file = fixtures_dir / "monthly_skill_demand.csv"
        source = CSVDataSource(pre_aggregated_path=pre_agg_file)

        demand = await source.get_monthly_skill_demand(["Python", "FastAPI"])
        assert len(demand) == 2
        assert len(demand["Python"]) == 12
        assert len(demand["FastAPI"]) == 12

    asyncio.run(_run())


def test_live_database_adapter_unconnected():
    async def _run():
        adapter = LiveDatabaseAdapter(db_session_or_url=None)
        assert adapter.get_source_label() == "live_database"
        totals = await adapter.get_total_jobs_by_month()
        assert totals == {}
        report = adapter.get_quality_report()
        assert len(report["errors"]) > 0

    asyncio.run(_run())


def test_validate_forecast_chronology_valid():
    from services.ai_forecasting.data.aggregator import validate_forecast_chronology
    from services.ai_forecasting.schemas import ForecastPoint

    obs = [HistoricalObservation(year_month="2026-10", count=100, total_jobs=400)]
    points = [
        ForecastPoint(year_month="2026-11", predicted_count=102.0),
        ForecastPoint(year_month="2026-12", predicted_count=105.0),
        ForecastPoint(year_month="2027-01", predicted_count=108.0),
    ]
    is_valid, issues = validate_forecast_chronology(obs, points, origin_month="2026-10")
    assert is_valid is True
    assert len(issues) == 0


def test_validate_forecast_chronology_detects_past_point():
    from services.ai_forecasting.data.aggregator import validate_forecast_chronology
    from services.ai_forecasting.schemas import ForecastPoint

    obs = [HistoricalObservation(year_month="2026-10", count=100, total_jobs=400)]
    # Point is in the past relative to origin (2026-05 <= 2026-10)
    invalid_points = [
        ForecastPoint(year_month="2026-05", predicted_count=90.0),
    ]
    is_valid, issues = validate_forecast_chronology(obs, invalid_points, origin_month="2026-10")
    assert is_valid is False
    assert any("not after forecast origin" in err for err in issues)


def test_validate_forecast_chronology_detects_gap():
    from services.ai_forecasting.data.aggregator import validate_forecast_chronology
    from services.ai_forecasting.schemas import ForecastPoint

    obs = [HistoricalObservation(year_month="2026-10", count=100, total_jobs=400)]
    # Skipped 2026-11 and jumped straight to 2026-12
    gap_points = [
        ForecastPoint(year_month="2026-12", predicted_count=105.0),
    ]
    is_valid, issues = validate_forecast_chronology(obs, gap_points, origin_month="2026-10")
    assert is_valid is False
    assert any("Non-consecutive forecast step" in err for err in issues)


def test_synthetic_demo_anchors_to_current_month():
    from datetime import datetime, timezone
    async def _run():
        source = SyntheticDemoDataSource(months_count=24, seed=42)
        demand = await source.get_monthly_skill_demand(["Python"])
        py_obs = demand["Python"]
        assert len(py_obs) == 24

        current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
        # Latest observation should match current calendar month
        assert py_obs[-1].year_month == current_ym

    asyncio.run(_run())


def test_synthetic_demo_custom_reference_month():
    async def _run():
        source = SyntheticDemoDataSource(reference_month="2025-06", months_count=12, seed=42)
        demand = await source.get_monthly_skill_demand(["Python"])
        py_obs = demand["Python"]
        assert len(py_obs) == 12
        assert py_obs[0].year_month == "2024-07"
        assert py_obs[-1].year_month == "2025-06"

    asyncio.run(_run())

