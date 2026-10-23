"""
Phase 4 Unit Tests: Skill Trajectory Classification.
"""

from services.ai_forecasting.classification.trajectory import classify_trajectory, TrajectoryClassifier
from services.ai_forecasting.data.synthetic_generator import SyntheticDemoDataSource
from services.ai_forecasting.schemas import HistoricalObservation


def test_classify_emerging_skill():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    fastapi_obs = series["FastAPI"]

    res = classify_trajectory("FastAPI", fastapi_obs)
    assert res.label == "Emerging"
    assert res.data_sufficiency is True
    assert res.growth_rate is not None and res.growth_rate > 0.20
    assert "Emerging skill" in res.summary


def test_classify_core_stable_skill():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    python_obs = series["Python"]

    res = classify_trajectory("Python", python_obs)
    assert res.label == "Core/Stable"
    assert res.data_sufficiency is True
    assert -0.15 <= res.growth_rate <= 0.20
    assert "Core/Stable skill" in res.summary


def test_classify_declining_skill():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    jquery_obs = series["jQuery"]

    res = classify_trajectory("jQuery", jquery_obs)
    assert res.label == "Declining"
    assert res.data_sufficiency is True
    assert res.growth_rate is not None and res.growth_rate < -0.15
    assert "Declining skill" in res.summary


def test_classify_niche_skill():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    rust_obs = series["Rust"]

    res = classify_trajectory("Rust", rust_obs)
    assert res.label == "Niche"
    assert res.data_sufficiency is True
    assert "Niche skill" in res.summary


def test_classify_insufficient_data():
    short_obs = [
        HistoricalObservation(year_month="2026-01", count=45, total_jobs=100),
        HistoricalObservation(year_month="2026-02", count=50, total_jobs=105),
    ]
    res = classify_trajectory("NewTech", short_obs)
    assert res.label == "Insufficient Data"
    assert res.data_sufficiency is False
    assert res.growth_rate is None
    assert "Insufficient historical data" in res.summary


def test_custom_thresholds():
    custom_classifier = TrajectoryClassifier(
        emerging_threshold=0.50,  # requires 50% growth
    )
    # 25% growth is emerging under standard (20%), but Core/Stable under strict (50%)
    obs = [
        HistoricalObservation(year_month="2025-01", count=40, total_jobs=200, demand_share=0.20),
        HistoricalObservation(year_month="2025-02", count=40, total_jobs=200, demand_share=0.20),
        HistoricalObservation(year_month="2025-03", count=50, total_jobs=200, demand_share=0.25),
        HistoricalObservation(year_month="2025-04", count=50, total_jobs=200, demand_share=0.25),
    ]
    res = custom_classifier.classify("SkillX", obs)
    assert res.label == "Core/Stable"


def test_python_trajectory_summary_contains_range_and_variation():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()
    python_obs = series["Python"]

    res = classify_trajectory("Python", python_obs)
    assert res.label == "Core/Stable"
    # Ensure summary explicitly details demand range and volatility instead of a naive zero-growth string
    assert "range:" in res.summary
    assert "std=" in res.summary
    assert "stability corridor" in res.summary
    assert f"ratio {res.recent_vs_earlier_ratio:.2f}" in res.summary

    # Verify counts range matches actual observations
    counts = [o.count for o in python_obs]
    expected_min = min(counts)
    expected_max = max(counts)
    assert f"{expected_min}-{expected_max}" in res.summary


def test_trajectory_summary_consistency_all_categories():
    source = SyntheticDemoDataSource(start_year=2024, months_count=24, seed=42)
    series, _ = source._generate()

    for skill, expected_label in [("FastAPI", "Emerging"), ("jQuery", "Declining"), ("Rust", "Niche"), ("Python", "Core/Stable")]:
        obs = series[skill]
        res = classify_trajectory(skill, obs)
        assert res.label == expected_label
        # All summaries must report range and threshold evidence
        assert "range:" in res.summary
        if expected_label != "Niche":
            assert f"ratio {res.recent_vs_earlier_ratio:.2f}" in res.summary

