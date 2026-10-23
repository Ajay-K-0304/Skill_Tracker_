"""
FastAPI router for AI Forecasting & Agentic Curriculum Recommendation.
Exposes POST /api/v1/forecast and POST /api/v1/curriculum/analyze endpoints.
Designed to be mounted directly into the central FastAPI gateway or run standalone.
"""

import logging
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from services.ai_forecasting.agents.workflow import (
    run_curriculum_workflow,
    run_curriculum_workflow_with_state,
)

logger = logging.getLogger(__name__)
from services.ai_forecasting.classification.trajectory import classify_trajectory
from services.ai_forecasting.config import settings
from services.ai_forecasting.data.aggregator import validate_forecast_chronology
from services.ai_forecasting.data.csv_loader import CSVDataSource
from services.ai_forecasting.data.db_adapter import LiveDatabaseAdapter
from services.ai_forecasting.data.interfaces import DataSourceAdapter
from services.ai_forecasting.data.synthetic_generator import SyntheticDemoDataSource
from services.ai_forecasting.forecasting.engine import SklearnTimeSeriesForecaster
from services.ai_forecasting.schemas import (
    ContractB,
    CurriculumAnalysisRequest,
    ForecastRequest,
    ForecastResponse,
    SkillForecastResult,
)
from services.ai_forecasting.syllabus.extractor import extract_text_from_file, SyllabusExtractorError

router = APIRouter(tags=["AI Forecasting & Curriculum Recommendation"])


def _get_data_adapter(
    source: str,
    csv_path: Optional[str] = None,
    reference_month: Optional[str] = None,
) -> DataSourceAdapter:
    """Factory helper to obtain the requested data source adapter."""
    if source == "synthetic_demo":
        return SyntheticDemoDataSource(reference_month=reference_month)
    elif source == "csv":
        fixtures_dir = Path(csv_path) if csv_path else settings.CSV_FIXTURES_DIR
        return CSVDataSource(fixtures_dir=fixtures_dir)
    elif source == "live_database":
        return LiveDatabaseAdapter()
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported data source '{source}'. Choose 'synthetic_demo', 'csv', or 'live_database'.",
    )


@router.get("/health", summary="Module Health Check")
async def health_check():
    """Returns the operational status of the Member 3 module."""
    return {
        "module": "services/ai_forecasting",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "default_horizon": settings.DEFAULT_FORECAST_HORIZON,
        "default_mode": settings.DATA_SOURCE_MODE,
    }


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Generate 6-to-12-Month Skill Demand Forecasts",
    description="Forecasts future monthly job market demand and classifies skill trajectories using time-series modeling.",
)
async def generate_skill_forecast(payload: ForecastRequest):
    """
    Endpoint A: Generates point predictions, prediction intervals, holdout MAE/RMSE,
    and trajectory classifications for requested skills across 6 or 12 month horizons.
    """
    adapter = _get_data_adapter(payload.source, payload.csv_path, payload.reference_month)
    series_map = await adapter.get_monthly_skill_demand(payload.skills)
    quality_report = adapter.get_quality_report()

    forecaster = SklearnTimeSeriesForecaster()
    skill_results: List[SkillForecastResult] = []
    global_warnings = list(quality_report.get("warnings", []))
    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")

    if not series_map:
        global_warnings.append(f"No historical skill series found in source '{payload.source}'.")

    overall_origin: Optional[str] = None

    for skill_name, observations in series_map.items():
        points, metrics, fc_status, limitations = forecaster.forecast(
            skill_name=skill_name,
            observations=observations,
            horizon_months=payload.horizon_months,
        )
        trajectory = classify_trajectory(
            skill_name=skill_name,
            observations=observations,
            forecast_points=points,
        )

        last_hist_month = observations[-1].year_month if observations else None
        fc_origin_month = last_hist_month

        if overall_origin is None and fc_origin_month:
            overall_origin = fc_origin_month

        # Validate chronology relative to origin and current execution date
        if observations and points:
            is_valid, chrono_issues = validate_forecast_chronology(
                observations=observations,
                forecast_points=points,
                origin_month=fc_origin_month,
                reference_month=current_ym,
            )
            for issue in chrono_issues:
                if "precedes reference month" in issue:
                    limitations.append(issue)
                    if issue not in global_warnings:
                        global_warnings.append(issue)
                elif not is_valid:
                    limitations.append(issue)

        skill_results.append(
            SkillForecastResult(
                skill_name=skill_name,
                horizon_months=payload.horizon_months,
                last_historical_month=last_hist_month,
                forecast_origin_month=fc_origin_month,
                historical=observations,
                forecast=points,
                trajectory=trajectory,
                metrics=metrics,
                status=fc_status,
                limitations=limitations,
            )
        )

    return ForecastResponse(
        skill_forecasts=skill_results,
        data_source=adapter.get_source_label(),
        forecast_origin=overall_origin,
        generated_at=datetime.now(timezone.utc).isoformat(),
        warnings=global_warnings,
    )


@router.post(
    "/curriculum/analyze",
    response_model=ContractB,
    summary="Analyze Syllabus against Role Benchmark (Contract B)",
    description="Evaluates academic course syllabi against industry job benchmarks and produces structured gap analysis and unit modifications.",
)
async def analyze_curriculum(payload: CurriculumAnalysisRequest):
    """
    Endpoint B (JSON pathway): Evaluates raw syllabus text against a target industry role.
    Returns Contract B response format.
    """
    if not payload.syllabus_text or not payload.syllabus_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="syllabus_text cannot be empty when submitting JSON payload.",
        )

    logger.info(
        f"Processing curriculum JSON analysis (target_role='{payload.target_role}', use_llm={payload.use_llm})"
    )
    try:
        contract_b, state = await run_curriculum_workflow_with_state(
            raw_text=payload.syllabus_text,
            target_role=payload.target_role,
            benchmark_override=payload.benchmark_skills,
            use_llm=payload.use_llm,
        )
        source = state.get("recommendation_source", "unknown")
        logger.info(
            f"Curriculum JSON analysis completed: course='{contract_b.course_name}', "
            f"target_role='{payload.target_role}', use_llm={payload.use_llm}, "
            f"recommendation_source='{source}', recommendations={len(contract_b.recommendations)}"
        )
        return contract_b
    except Exception as e:
        logger.error(f"Curriculum analysis workflow failed: {str(e)[:200]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Curriculum analysis workflow failed: {str(e)}",
        )


@router.post(
    "/curriculum/analyze/upload",
    response_model=ContractB,
    summary="Upload Syllabus Document (PDF/Text) for Gap Analysis",
    description="Accepts PDF or text file uploads, extracts course structure, and generates Contract B curriculum recommendations.",
)
async def analyze_curriculum_upload(
    file: UploadFile = File(..., description="Syllabus document (PDF or Text)"),
    target_role: str = Form("Cloud Engineer", description="Target industry role for benchmark comparison"),
    use_llm: bool = Form(False, description="Whether to invoke local LLM for recommendation synthesis"),
    benchmark_override: Optional[str] = Form(None, description="Optional JSON array or comma-separated benchmark skills"),
):
    """
    Endpoint B (File upload pathway): Ingests PDF or text file, validates file integrity,
    and runs the LangGraph recommendation workflow.
    """
    logger.info(
        f"Processing curriculum upload analysis: filename='{file.filename}', "
        f"content_type='{file.content_type}', target_role='{target_role}', use_llm={use_llm}"
    )
    content = await file.read()

    try:
        raw_text = extract_text_from_file(
            content=content,
            filename=file.filename,
            content_type=file.content_type,
        )
    except SyllabusExtractorError as e:
        logger.warning(f"Curriculum upload extraction error: {str(e)[:150]}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    override_skills = None
    if benchmark_override:
        benchmark_str = benchmark_override.strip()
        if benchmark_str.startswith("["):
            try:
                override_skills = json.loads(benchmark_str)
            except Exception:
                override_skills = [s.strip() for s in benchmark_str.split(",") if s.strip()]
        else:
            override_skills = [s.strip() for s in benchmark_str.split(",") if s.strip()]

    try:
        contract_b, state = await run_curriculum_workflow_with_state(
            raw_text=raw_text,
            target_role=target_role,
            benchmark_override=override_skills,
            use_llm=use_llm,
        )
        source = state.get("recommendation_source", "unknown")
        logger.info(
            f"Curriculum upload analysis completed: course='{contract_b.course_name}', "
            f"target_role='{target_role}', use_llm={use_llm}, "
            f"recommendation_source='{source}', recommendations={len(contract_b.recommendations)}"
        )
        return contract_b
    except Exception as e:
        logger.error(f"Curriculum upload analysis workflow failed: {str(e)[:200]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Curriculum analysis workflow failed: {str(e)}",
        )
