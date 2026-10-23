"""
Pydantic data schemas and contracts for Member 3's AI Forecasting & Curriculum Recommendation service.
Includes Contract B specification, forecasting models, and internal workflow state schemas.
"""

from datetime import datetime, timezone
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Trajectory and Forecast Schemas
# ---------------------------------------------------------------------------

TrajectoryLabelType = Literal["Emerging", "Core/Stable", "Declining", "Niche", "Insufficient Data"]
PriorityType = Literal["High", "Medium", "Low"]
DataSourceType = Literal["synthetic_demo", "csv", "live_database"]


class HistoricalObservation(BaseModel):
    """Monthly historical skill observation record."""
    year_month: str = Field(..., description="Observation month in YYYY-MM format", pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    count: int = Field(..., ge=0, description="Unique job postings requesting this skill in this month")
    total_jobs: int = Field(..., ge=0, description="Total distinct job postings analyzed in this month")
    demand_share: Optional[float] = Field(None, ge=0.0, le=1.0, description="Skill count / total_jobs ratio")


class ForecastPoint(BaseModel):
    """Projected future demand point for a skill."""
    year_month: str = Field(..., description="Projected month in YYYY-MM format", pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    predicted_count: float = Field(..., ge=0.0, description="Estimated non-negative job demand count")
    lower_bound: Optional[float] = Field(None, ge=0.0, description="Lower prediction interval (non-negative)")
    upper_bound: Optional[float] = Field(None, ge=0.0, description="Upper prediction interval")


class TrajectoryResult(BaseModel):
    """Classification of a skill's market trajectory with supporting evidence."""
    label: TrajectoryLabelType = Field(..., description="Categorical trajectory label")
    growth_rate: Optional[float] = Field(None, description="Calculated trend growth rate over historical observations")
    recent_vs_earlier_ratio: Optional[float] = Field(None, description="Ratio of recent period demand to earlier period")
    summary: str = Field(..., description="Human-readable explanation of trajectory evidence and thresholds")
    data_sufficiency: bool = Field(..., description="True if sufficient time-series history supported classification")


class EvaluationMetrics(BaseModel):
    """Time-series holdout evaluation metrics (MAE and RMSE)."""
    mae: Optional[float] = Field(None, description="Mean Absolute Error on holdout sample")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error on holdout sample")
    evaluation_samples: Optional[int] = Field(None, description="Number of holdout months evaluated")
    notes: Optional[str] = Field(None, description="Explanation or limitations regarding sample size")


class SkillForecastResult(BaseModel):
    """Complete forecast package for a single skill."""
    skill_name: str = Field(..., description="Canonical name of the skill")
    horizon_months: int = Field(..., description="Forecast horizon in months (6 or 12)")
    last_historical_month: Optional[str] = Field(None, description="Latest month in the historical observation series in YYYY-MM format")
    forecast_origin_month: Optional[str] = Field(None, description="Anchor month from which future projections are generated in YYYY-MM format")
    historical: List[HistoricalObservation] = Field(default_factory=list, description="Historical monthly series")
    forecast: List[ForecastPoint] = Field(default_factory=list, description="Future monthly projections")
    trajectory: TrajectoryResult = Field(..., description="Deterministic trajectory classification")
    metrics: Optional[EvaluationMetrics] = Field(None, description="Model evaluation metrics when sample permits")
    status: str = Field("success", description="Status string: success, insufficient_data, or warning")
    limitations: List[str] = Field(default_factory=list, description="Identified data or model limitations")


class ForecastRequest(BaseModel):
    """Input payload for POST /api/v1/forecast endpoint."""
    source: DataSourceType = Field("synthetic_demo", description="Data source to query")
    horizon_months: Literal[6, 12] = Field(6, description="Forecast horizon (6 or 12 months)")
    skills: Optional[List[str]] = Field(None, description="Optional list of skills to filter/forecast")
    csv_path: Optional[str] = Field(None, description="Optional custom CSV path for CSV mode")
    reference_month: Optional[str] = Field(None, description="Optional reference origin month override (YYYY-MM) for synthetic generation or backtesting")


class ForecastResponse(BaseModel):
    """Output payload for POST /api/v1/forecast endpoint."""
    skill_forecasts: List[SkillForecastResult] = Field(..., description="List of forecasted skills")
    data_source: str = Field(..., description="Data source used (synthetic_demo, csv, or live_database)")
    forecast_origin: Optional[str] = Field(None, description="Primary forecast origin month in YYYY-MM format")
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp")
    warnings: List[str] = Field(default_factory=list, description="Global warnings regarding data quality or coverage")


# ---------------------------------------------------------------------------
# Contract B: Syllabus Gap Analysis & Recommendations
# ---------------------------------------------------------------------------

class RecommendationItem(BaseModel):
    """Concrete actionable curriculum modification proposal."""
    unit: str = Field(..., description="Affected course unit/module (e.g., 'Unit 3: Service Communication')")
    action: str = Field(..., description="Concrete modification, lab, or topic introduction")
    priority: PriorityType = Field(..., description="Urgency of recommendation (High, Medium, Low)")


class ContractB(BaseModel):
    """
    Contract B: Detailed schema shared with Member 4 (Dashboard) and university evaluators.
    Matches exact specification:
    {
      "course_name": "...",
      "coverage_score": 71.4,
      "covered_skills": [...],
      "missing_skills": [...],
      "recommendations": [...]
    }
    """
    course_name: str = Field(..., description="Name of the course parsed from syllabus or provided")
    coverage_score: float = Field(..., ge=0.0, le=100.0, description="Percentage of benchmark skills covered (0-100)")
    covered_skills: List[str] = Field(default_factory=list, description="List of canonical benchmark skills present in syllabus")
    missing_skills: List[str] = Field(default_factory=list, description="List of canonical benchmark skills absent from syllabus")
    recommendations: List[RecommendationItem] = Field(default_factory=list, description="List of structured recommendations")

    @field_validator("coverage_score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        return round(float(v), 2)


class SyllabusUnit(BaseModel):
    """Extracted unit or module from a syllabus."""
    unit_title: str = Field(..., description="Unit title or heading")
    topics: List[str] = Field(default_factory=list, description="Topics taught within this unit")
    learning_objectives: List[str] = Field(default_factory=list, description="Stated learning objectives if present")


class ParsedSyllabus(BaseModel):
    """Intermediate structured extraction from a syllabus document."""
    course_name: str = Field(..., description="Identified course title")
    units: List[SyllabusUnit] = Field(default_factory=list, description="Extracted units and modules")
    extracted_skills: List[str] = Field(default_factory=list, description="Canonical skills recognized in syllabus text")
    raw_text_length: int = Field(0, description="Character count of extracted raw text")
    extraction_notes: Optional[str] = Field(None, description="Notes on extraction quality, OCR alerts, etc.")


class CurriculumAnalysisRequest(BaseModel):
    """Input payload for POST /api/v1/curriculum/analyze endpoint when sent as JSON."""
    syllabus_text: Optional[str] = Field(None, description="Raw syllabus text if not uploading a file")
    target_role: str = Field("Cloud Engineer", description="Target industry role for benchmark comparison")
    benchmark_skills: Optional[List[str]] = Field(None, description="Explicit benchmark skills override for testing")
    use_llm: bool = Field(False, description="Whether to invoke local LLM for recommendation synthesis")
