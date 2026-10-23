"""
Phase 8 Unit Tests: API Endpoints, OpenAPI generation, File Uploads, and Output Validation.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from starlette.testclient import TestClient
from services.ai_forecasting.standalone_app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["module"] == "services/ai_forecasting"


def test_forecast_endpoint_synthetic_demo_6_months(client):
    payload = {
        "source": "synthetic_demo",
        "horizon_months": 6,
    }
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["data_source"] == "synthetic_demo"
    assert "generated_at" in data
    assert "forecast_origin" in data
    assert len(data["skill_forecasts"]) >= 4

    # Index by skill name
    by_skill = {item["skill_name"]: item for item in data["skill_forecasts"]}
    assert "Python" in by_skill
    assert "FastAPI" in by_skill
    assert "jQuery" in by_skill
    assert "Rust" in by_skill

    # Verify origin and last historical month identification
    py = by_skill["Python"]
    assert py["last_historical_month"] is not None
    assert py["forecast_origin_month"] == py["last_historical_month"]
    origin = py["forecast_origin_month"]

    # Verify forecast points are chronologically AFTER the origin
    assert len(py["forecast"]) == 6
    assert py["forecast"][0]["year_month"] > origin
    for fc in py["forecast"]:
        assert fc["year_month"] > origin

    assert by_skill["Python"]["trajectory"]["label"] == "Core/Stable"
    assert by_skill["FastAPI"]["trajectory"]["label"] == "Emerging"
    assert by_skill["jQuery"]["trajectory"]["label"] == "Declining"
    assert by_skill["Rust"]["trajectory"]["label"] == "Niche"

    # Verify non-negativity
    for fc in by_skill["jQuery"]["forecast"]:
        assert fc["predicted_count"] >= 0.0
        assert fc["lower_bound"] >= 0.0


def test_forecast_endpoint_12_months_and_filter(client):
    payload = {
        "source": "synthetic_demo",
        "horizon_months": 12,
        "skills": ["FastAPI"],
    }
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["skill_forecasts"]) == 1
    fastapi_res = data["skill_forecasts"][0]
    assert fastapi_res["skill_name"] == "FastAPI"
    assert len(fastapi_res["forecast"]) == 12
    origin = fastapi_res["forecast_origin_month"]
    for pt in fastapi_res["forecast"]:
        assert pt["year_month"] > origin


def test_forecast_endpoint_detects_past_origin_warning(client):
    # Requesting a historical reference month in 2025-06 while generated_at is in 2026-10
    payload = {
        "source": "synthetic_demo",
        "horizon_months": 6,
        "reference_month": "2025-06",
    }
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["forecast_origin"] == "2025-06"
    assert any("precedes reference month" in w for w in data["warnings"])
    py = data["skill_forecasts"][0]
    assert py["forecast_origin_month"] == "2025-06"
    assert py["forecast"][0]["year_month"] == "2025-07"
    assert py["forecast"][-1]["year_month"] == "2025-12"


def test_forecast_endpoint_csv_source(client):
    payload = {
        "source": "csv",
        "horizon_months": 6,
    }
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["data_source"] == "csv"
    assert any("Historical timespan is too short" in w for w in data["warnings"])


def test_curriculum_analyze_json_contract_b(client):
    txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
    with open(txt_path, "r", encoding="utf-8") as f:
        syllabus_text = f.read()

    payload = {
        "syllabus_text": syllabus_text,
        "target_role": "Cloud Engineer",
        "use_llm": False,
    }
    response = client.post("/api/v1/curriculum/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Strictly verify Contract B schema
    assert "course_name" in data
    assert "coverage_score" in data
    assert "covered_skills" in data
    assert "missing_skills" in data
    assert "recommendations" in data

    assert data["course_name"] == "Distributed Systems & Cloud Computing"
    assert 0.0 <= data["coverage_score"] <= 100.0
    assert "Docker" in data["covered_skills"]
    assert "Kubernetes" in data["missing_skills"]

    # Verify recommendation item structure
    assert len(data["recommendations"]) > 0
    first_rec = data["recommendations"][0]
    assert "unit" in first_rec
    assert "action" in first_rec
    assert first_rec["priority"] in ["High", "Medium", "Low"]


def test_curriculum_analyze_upload_pdf(client):
    pdf_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.pdf"
    with open(pdf_path, "rb") as f:
        files = {"file": ("sample_syllabus.pdf", f, "application/pdf")}
        data = {
            "target_role": "Cloud Engineer",
            "use_llm": "false",
        }
        response = client.post("/api/v1/curriculum/analyze/upload", files=files, data=data)

    assert response.status_code == 200
    res_data = response.json()
    assert "Distributed Systems" in res_data["course_name"]
    assert len(res_data["recommendations"]) > 0


def test_curriculum_analyze_upload_empty_file_fails(client):
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/api/v1/curriculum/analyze/upload", files=files, data={"target_role": "Cloud Engineer"})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_curriculum_analyze_upload_pdf_with_groq_mocked(client, caplog):
    import json
    import logging
    from unittest.mock import AsyncMock, MagicMock, patch

    caplog.set_level(logging.INFO)
    pdf_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.pdf"

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "recommendations": [
            {
                "unit": "Unit 2: Containerization and Infrastructure",
                "action": "Integrate production Kubernetes deployments and services lab.",
                "priority": "High",
            },
            {
                "unit": "Unit 3: Service Communication",
                "action": "Add hands-on gRPC streaming practical assignment.",
                "priority": "Medium",
            },
        ]
    })
    mock_completion.choices = [mock_choice]

    with patch("groq.AsyncGroq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_groq_cls.return_value = mock_instance

        with patch("services.ai_forecasting.config.settings.GROQ_API_KEY", "gsk_mock_api_key_test"):
            with open(pdf_path, "rb") as f:
                files = {"file": ("sample_syllabus.pdf", f, "application/pdf")}
                data = {"target_role": "Cloud Engineer", "use_llm": "true"}
                response = client.post("/api/v1/curriculum/analyze/upload", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["course_name"] == "Distributed Systems & Cloud Computing"
    assert len(res_json["recommendations"]) == 2
    assert "Kubernetes deployments" in res_json["recommendations"][0]["action"]

    all_logs = caplog.text
    assert "Initiating LLM curriculum recommendation synthesis" in all_logs
    assert "Successfully synthesized 2 recommendations via Groq API" in all_logs
    assert "recommendation_source='groq'" in all_logs
    assert "gsk_mock_api_key_test" not in all_logs


def test_curriculum_analyze_upload_pdf_with_groq_failure_fallback_mocked(client, caplog):
    import logging
    from unittest.mock import AsyncMock, MagicMock, patch

    caplog.set_level(logging.INFO)
    pdf_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.pdf"

    secret_key = "gsk_mock_secret_error_token_456"

    with patch("groq.AsyncGroq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(
            side_effect=RuntimeError(f"Groq API 500 Internal Error token {secret_key}")
        )
        mock_groq_cls.return_value = mock_instance

        with patch("services.ai_forecasting.config.settings.GROQ_API_KEY", secret_key):
            with open(pdf_path, "rb") as f:
                files = {"file": ("sample_syllabus.pdf", f, "application/pdf")}
                data = {"target_role": "Cloud Engineer", "use_llm": "true"}
                response = client.post("/api/v1/curriculum/analyze/upload", files=files, data=data)

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["course_name"] == "Distributed Systems & Cloud Computing"
    # Fallback recommendations are generated
    assert len(res_json["recommendations"]) > 0

    all_logs = caplog.text
    assert "LLM synthesis failed for provider 'groq'" in all_logs
    assert "Applying deterministic fallback recommendation rules" in all_logs
    assert "recommendation_source='deterministic_fallback'" in all_logs
    assert secret_key not in all_logs



def test_openapi_schema_generation(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    openapi = response.json()
    paths = openapi.get("paths", {})
    assert "/api/v1/forecast" in paths
    assert "/api/v1/curriculum/analyze" in paths
    assert "/api/v1/curriculum/analyze/upload" in paths
    assert "/api/v1/health" in paths


def test_forecast_endpoint_synthetic_demo_12_months_full_suite(client):
    payload = {
        "source": "synthetic_demo",
        "horizon_months": 12,
    }
    response = client.post("/api/v1/forecast", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["data_source"] == "synthetic_demo"
    assert len(data["skill_forecasts"]) == 4

    by_skill = {item["skill_name"]: item for item in data["skill_forecasts"]}

    # Issue 1: Python trajectory consistency and range details
    py = by_skill["Python"]
    assert py["trajectory"]["label"] == "Core/Stable"
    assert "range:" in py["trajectory"]["summary"]
    assert "stability corridor" in py["trajectory"]["summary"]
    assert -0.15 <= py["trajectory"]["growth_rate"] <= 0.20

    # Issue 2: Extrapolation, non-negative clipping, and prediction intervals
    for skill_name, item in by_skill.items():
        assert len(item["forecast"]) == 12
        assert len(item["historical"]) == 24
        origin = item["forecast_origin_month"]

        for pt in item["forecast"]:
            assert pt["year_month"] > origin
            assert pt["predicted_count"] >= 0.0
            assert pt["lower_bound"] >= 0.0
            assert pt["upper_bound"] >= pt["predicted_count"]
            assert pt["upper_bound"] >= pt["lower_bound"]

    # Rising FastAPI reaches high volume and interval widens
    fa = by_skill["FastAPI"]
    assert fa["forecast"][-1]["predicted_count"] > fa["forecast"][0]["predicted_count"]

    # Declining jQuery hits 0 by end of horizon
    jq = by_skill["jQuery"]
    assert jq["forecast"][-1]["predicted_count"] == 0.0

    # Issue 3: Evaluation quality (6-month holdout for 24-month series)
    for skill_name, item in by_skill.items():
        met = item["metrics"]
        assert met["evaluation_samples"] == 6
        assert "6 months test" in met["notes"]
        assert "18 months train" in met["notes"]


def test_curriculum_analyze_endpoint_groq_mocked(client):
    import json
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({
        "recommendations": [
            {
                "unit": "Unit 2: Containerization and Infrastructure",
                "action": "Integrate hands-on Kubernetes orchestration and Helm charts.",
                "priority": "High",
            }
        ]
    })
    mock_completion.choices = [mock_choice]

    with patch("groq.AsyncGroq") as mock_groq_cls:
        mock_instance = MagicMock()
        mock_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_groq_cls.return_value = mock_instance

        with patch("services.ai_forecasting.config.settings.GROQ_API_KEY", "gsk_test_api_key"):
            txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
            with open(txt_path, "r", encoding="utf-8") as f:
                syllabus_text = f.read()

            payload = {
                "syllabus_text": syllabus_text,
                "target_role": "Cloud Engineer",
                "use_llm": True,
            }
            response = client.post("/api/v1/curriculum/analyze", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["course_name"] == "Distributed Systems & Cloud Computing"
            assert len(data["recommendations"]) >= 1
            assert data["recommendations"][0]["unit"] == "Unit 2: Containerization and Infrastructure"


def test_curriculum_analyze_endpoint_missing_key_fallback(client):
    # When use_llm=true but GROQ_API_KEY is not set, must gracefully fall back to deterministic
    with patch("services.ai_forecasting.config.settings.GROQ_API_KEY", None):
        txt_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample_syllabus.txt"
        with open(txt_path, "r", encoding="utf-8") as f:
            syllabus_text = f.read()

        payload = {
            "syllabus_text": syllabus_text,
            "target_role": "Cloud Engineer",
            "use_llm": True,
        }
        response = client.post("/api/v1/curriculum/analyze", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["course_name"] == "Distributed Systems & Cloud Computing"
        assert len(data["recommendations"]) >= 2
        assert "Kubernetes" in data["missing_skills"]


