"""
Phase 6 Unit Tests: Target-Role Skill Benchmark and Gap Analysis.
"""

import pytest
from services.ai_forecasting.curriculum.gap_analyzer import analyze_skill_gap
from services.ai_forecasting.curriculum.benchmarks import get_target_role_benchmark


def test_perfect_coverage():
    syllabus = ["Docker", "Linux", "REST APIs", "Kubernetes", "gRPC", "FastAPI"]
    benchmark = ["Docker", "Linux", "REST APIs", "Kubernetes", "gRPC", "FastAPI"]

    score, covered, missing = analyze_skill_gap(syllabus, benchmark)
    assert score == 100.0
    assert len(covered) == 6
    assert len(missing) == 0


def test_partial_coverage_contract_b_benchmark():
    # 5 covered out of 7 skills -> (5/7)*100 = 71.4%
    benchmark = ["Docker", "Linux", "REST APIs", "Kubernetes", "gRPC", "FastAPI", "Python"]
    syllabus = ["Docker", "Linux", "REST APIs", "Python", "FastAPI"]

    score, covered, missing = analyze_skill_gap(syllabus, benchmark)
    assert score == 71.4
    assert len(covered) == 5
    assert sorted(missing) == ["Kubernetes", "gRPC"]


def test_zero_coverage():
    benchmark = ["Kubernetes", "gRPC", "AWS"]
    syllabus = ["Photoshop", "Figma", "Marketing"]

    score, covered, missing = analyze_skill_gap(syllabus, benchmark)
    assert score == 0.0
    assert len(covered) == 0
    assert sorted(missing) == ["AWS", "Kubernetes", "gRPC"]


def test_duplicate_skills_normalization():
    # Duplicates in both lists with varying casing
    benchmark = ["Docker", "docker", "LINUX", "Linux", "Kubernetes"]  # 3 unique
    syllabus = ["DOCKER", "docker", "linux"]  # 2 matched

    score, covered, missing = analyze_skill_gap(syllabus, benchmark)
    assert score == 66.7
    assert sorted(covered) == ["Docker", "Linux"]
    assert missing == ["Kubernetes"]


def test_empty_benchmark_error():
    with pytest.raises(ValueError, match="at least one skill"):
        analyze_skill_gap(["Python"], [])


def test_get_target_role_benchmark():
    skills, source = get_target_role_benchmark("Cloud Engineer")
    assert "Docker" in skills
    assert source == "mock_role_benchmark"

    # Override
    custom = ["Go", "Terraform"]
    skills_override, source_override = get_target_role_benchmark("Cloud Engineer", override=custom)
    assert skills_override == ["Go", "Terraform"]
    assert source_override == "user_override"
