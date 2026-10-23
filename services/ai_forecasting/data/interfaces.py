"""
Abstract base class and protocol for data sources.
Allows swapping between Live Database, CSV files, and Synthetic Demo data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from services.ai_forecasting.schemas import HistoricalObservation


class DataSourceAdapter(ABC):
    """Abstract interface for monthly skill-demand data sources."""

    @abstractmethod
    async def get_monthly_skill_demand(
        self, skills: Optional[List[str]] = None
    ) -> Dict[str, List[HistoricalObservation]]:
        """
        Retrieves monthly time-series for requested skills (or all available skills).
        Returns mapping from canonical skill_name to sorted list of HistoricalObservation objects.
        """
        pass

    @abstractmethod
    async def get_total_jobs_by_month(self) -> Dict[str, int]:
        """
        Retrieves total distinct job postings aggregated per month ('YYYY-MM' -> count).
        Must be computed independently from skill-linked counts.
        """
        pass

    @abstractmethod
    def get_source_label(self) -> str:
        """Returns the identifier label of the data source ('live_database', 'csv', 'synthetic_demo')."""
        pass

    @abstractmethod
    def get_quality_report(self) -> Dict[str, any]:
        """Returns metadata about data completeness, date ranges, and warnings."""
        pass
