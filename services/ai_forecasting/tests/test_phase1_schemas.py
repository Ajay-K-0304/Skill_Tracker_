"""
Phase 1 Unit Tests: Verification of configuration loading and schema validation.
"""

import pytest
from pydantic import ValidationError
from services.ai_forecasting.config import Settings
from services.ai_forecasting.schemas import (
    ContractB,
    RecommendationItem,
    ForecastRequest,
    ForecastResponse,
    HistoricalObservation,
    ForecastPoint,
    TrajectoryResult,
    SkillForecastResult,
)


def test_settings_initialization():
    settings = Settings()
    assert settings.DEFAULT_FORECAST_HORIZON == 6
    assert settings.DATA_SOURCE_MODE in ["synthetic_demo", "csv", "live_database"]
    assert settings.LOCAL_LLM_PROVIDER in ["groq", "ollama", "openai_compatible", "mock"]


def test_contract_b_valid():
    contract = ContractB(
        course_name="Distributed Systems & Cloud Computing",
        coverage_score=71.4,
        covered_skills=["Docker", "Linux", "REST APIs"],
        missing_skills=["Kubernetes", "gRPC", "FastAPI"],
        recommendations=[
            RecommendationItem(
                unit="Unit 3: Service Communication",
                action="Introduce gRPC and Protocol Buffers alongside REST APIs.",
                priority="High",
            )
        ],
    )
    assert contract.course_name == "Distributed Systems & Cloud Computing"
    assert contract.coverage_score == 71.4
    assert len(contract.covered_skills) == 3
    assert len(contract.missing_skills) == 3
    assert len(contract.recommendations) == 1
    assert contract.recommendations[0].priority == "High"


def test_contract_b_invalid_score():
    with pytest.raises(ValidationError):
        ContractB(
            course_name="Invalid Score Course",
            coverage_score=150.0,  # Invalid: must be <= 100.0
            covered_skills=[],
            missing_skills=[],
            recommendations=[],
        )


def test_historical_observation_validation():
    # Valid
    obs = HistoricalObservation(year_month="2026-05", count=45, total_jobs=120, demand_share=0.375)
    assert obs.count == 45
    assert obs.total_jobs == 120

    # Negative count should fail
    with pytest.raises(ValidationError):
        HistoricalObservation(year_month="2026-05", count=-5, total_jobs=100)

    # Invalid year_month format
    with pytest.raises(ValidationError):
        HistoricalObservation(year_month="2026-13", count=10, total_jobs=100)


def test_forecast_request_defaults():
    req = ForecastRequest()
    assert req.source == "synthetic_demo"
    assert req.horizon_months == 6
    assert req.skills is None


def test_forecast_response_schema():
    skill_result = SkillForecastResult(
        skill_name="Python",
        horizon_months=6,
        historical=[HistoricalObservation(year_month="2026-01", count=10, total_jobs=100)],
        forecast=[ForecastPoint(year_month="2026-02", predicted_count=12.0, lower_bound=10.0, upper_bound=14.0)],
        trajectory=TrajectoryResult(
            label="Core/Stable",
            growth_rate=0.05,
            recent_vs_earlier_ratio=1.05,
            summary="Demand is steady with moderate job count.",
            data_sufficiency=True,
        ),
        status="success",
        limitations=[],
    )
    response = ForecastResponse(
        skill_forecasts=[skill_result],
        data_source="synthetic_demo",
    )
    assert len(response.skill_forecasts) == 1
    assert response.skill_forecasts[0].skill_name == "Python"
