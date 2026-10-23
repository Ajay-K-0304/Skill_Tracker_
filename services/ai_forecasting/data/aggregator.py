"""
Monthly skill-demand aggregation, missing-month zero-filling, and validation logic.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from services.ai_forecasting.schemas import ForecastPoint, HistoricalObservation


def parse_year_month(ym: str) -> Tuple[int, int]:
    """Parses 'YYYY-MM' into (year, month)."""
    parts = ym.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid year_month format '{ym}', expected 'YYYY-MM'")
    return int(parts[0]), int(parts[1])


def format_year_month(year: int, month: int) -> str:
    """Formats year and month into 'YYYY-MM'."""
    return f"{year:04d}-{month:02d}"


def add_months(ym: str, n: int) -> str:
    """Adds n months to 'YYYY-MM' string."""
    year, month = parse_year_month(ym)
    total_months = year * 12 + (month - 1) + n
    new_year = total_months // 12
    new_month = (total_months % 12) + 1
    return format_year_month(new_year, new_month)


def generate_month_range(start_ym: str, end_ym: str) -> List[str]:
    """Generates continuous inclusive list of 'YYYY-MM' between start and end."""
    start_y, start_m = parse_year_month(start_ym)
    end_y, end_m = parse_year_month(end_ym)

    start_idx = start_y * 12 + (start_m - 1)
    end_idx = end_y * 12 + (end_m - 1)

    if start_idx > end_idx:
        return []

    result = []
    for idx in range(start_idx, end_idx + 1):
        y = idx // 12
        m = (idx % 12) + 1
        result.append(format_year_month(y, m))
    return result


def detect_missing_months(present_months: List[str]) -> List[str]:
    """Identifies any gaps in months between min and max observed months."""
    if not present_months:
        return []

    sorted_months = sorted(set(present_months))
    full_range = generate_month_range(sorted_months[0], sorted_months[-1])
    present_set = set(sorted_months)
    return [m for m in full_range if m not in present_set]


def fill_missing_months(
    observations: List[HistoricalObservation],
    total_jobs_by_month: Optional[Dict[str, int]] = None,
    zero_fill: bool = True,
) -> List[HistoricalObservation]:
    """
    Zero-fills absent months between the earliest and latest observation for a skill.
    Preserves actual counts where available, and safely computes demand_share.
    """
    if not observations:
        return []

    # Sort existing observations by year_month
    obs_map: Dict[str, HistoricalObservation] = {obs.year_month: obs for obs in observations}
    sorted_months = sorted(obs_map.keys())
    full_range = generate_month_range(sorted_months[0], sorted_months[-1])

    filled: List[HistoricalObservation] = []
    for ym in full_range:
        if ym in obs_map:
            obs = obs_map[ym]
            # Ensure demand_share is computed if total_jobs > 0
            demand_share = obs.demand_share
            if demand_share is None and obs.total_jobs > 0:
                demand_share = round(obs.count / obs.total_jobs, 4)
            filled.append(
                HistoricalObservation(
                    year_month=ym,
                    count=obs.count,
                    total_jobs=obs.total_jobs,
                    demand_share=demand_share,
                )
            )
        elif zero_fill:
            # Absent month: count is zero
            m_total = total_jobs_by_month.get(ym, 0) if total_jobs_by_month else 0
            filled.append(
                HistoricalObservation(
                    year_month=ym,
                    count=0,
                    total_jobs=m_total,
                    demand_share=0.0 if m_total > 0 else None,
                )
            )

    return filled


def validate_monthly_observations(observations: List[HistoricalObservation]) -> List[str]:
    """
    Performs data quality checks on historical observations:
    - Checks for duplicate months
    - Checks for negative counts
    - Checks for count > total_jobs
    - Checks time-series length
    Returns a list of warning/error messages.
    """
    warnings: List[str] = []
    if not observations:
        warnings.append("No historical observations provided.")
        return warnings

    seen_months = set()
    for obs in observations:
        if obs.year_month in seen_months:
            warnings.append(f"Duplicate observation detected for month: {obs.year_month}")
        seen_months.add(obs.year_month)

        if obs.count < 0:
            warnings.append(f"Negative skill count for month {obs.year_month}: {obs.count}")

        if obs.total_jobs < 0:
            warnings.append(f"Negative total_jobs for month {obs.year_month}: {obs.total_jobs}")

        if obs.total_jobs > 0 and obs.count > obs.total_jobs:
            warnings.append(
                f"Skill count ({obs.count}) exceeds total jobs ({obs.total_jobs}) in month {obs.year_month}"
            )

    if len(observations) < 4:
        warnings.append(
            f"Insufficient history: only {len(observations)} months available (minimum 4 required for baseline forecasting)."
        )

    return warnings


def validate_forecast_chronology(
    observations: List[HistoricalObservation],
    forecast_points: List[ForecastPoint],
    origin_month: Optional[str] = None,
    reference_month: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Validates chronological integrity of forecast points relative to the forecast origin.
    Verifies that:
    1. Forecast origin is determined from the latest historical observation if not explicitly passed.
    2. Every forecast point's year_month is strictly chronologically AFTER the forecast origin.
    3. Forecast points are continuous and strictly monotonic (each point is exactly 1 month after previous).
    4. Detects whether the forecast origin or forecast points lie in the past relative to reference_month.

    Returns:
        Tuple of (is_valid: bool, issues: List[str])
    """
    issues: List[str] = []
    if not observations:
        issues.append("Cannot establish forecast chronology without historical observations.")
        return False, issues

    sorted_obs = sorted(observations, key=lambda x: x.year_month)
    last_hist = sorted_obs[-1].year_month
    origin = origin_month or last_hist

    if not forecast_points:
        return True, issues

    # 1. Verify every forecast point is strictly after the origin
    for idx, pt in enumerate(forecast_points, start=1):
        expected_ym = add_months(origin, idx)
        if pt.year_month <= origin:
            issues.append(
                f"Invalid forecast chronology: forecast point '{pt.year_month}' is not after "
                f"forecast origin '{origin}'."
            )
        if pt.year_month != expected_ym:
            issues.append(
                f"Non-consecutive forecast step at index {idx}: expected '{expected_ym}', got '{pt.year_month}'."
            )

    # 2. Check relation to reference_month (e.g. current calendar month)
    if reference_month and origin < reference_month:
        issues.append(
            f"Forecast origin '{origin}' precedes reference month '{reference_month}'. "
            f"Forecast points project from historical observation end, not current date."
        )

    is_valid = not any("Invalid" in err or "Non-consecutive" in err for err in issues)
    return is_valid, issues
