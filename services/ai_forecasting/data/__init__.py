"""
Data preparation, aggregation, and data-source adapters.
"""

from services.ai_forecasting.data.interfaces import DataSourceAdapter
from services.ai_forecasting.data.csv_loader import CSVDataSource
from services.ai_forecasting.data.synthetic_generator import SyntheticDemoDataSource
from services.ai_forecasting.data.aggregator import (
    fill_missing_months,
    detect_missing_months,
    validate_monthly_observations,
    validate_forecast_chronology,
)

__all__ = [
    "DataSourceAdapter",
    "CSVDataSource",
    "SyntheticDemoDataSource",
    "fill_missing_months",
    "detect_missing_months",
    "validate_monthly_observations",
    "validate_forecast_chronology",
]
