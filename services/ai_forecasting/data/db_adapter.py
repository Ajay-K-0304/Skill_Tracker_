"""
Live Database Adapter for Member 1's PostgreSQL schema.
Dynamically derives monthly aggregates without requiring a pre-computed skill_monthly_stats table.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
from services.ai_forecasting.data.interfaces import DataSourceAdapter
from services.ai_forecasting.data.aggregator import fill_missing_months
from services.ai_forecasting.schemas import HistoricalObservation


class LiveDatabaseAdapter(DataSourceAdapter):
    """
    Adapter that executes dynamic monthly aggregation against Member 1's schema:
    job_postings -> job_skills -> skill_taxonomy
    """

    def __init__(self, db_session_or_url: Optional[Any] = None):
        self.db_target = db_session_or_url
        self._quality_report: Dict[str, Any] = {
            "source": "live_database",
            "connected": False,
            "warnings": [],
            "errors": [],
        }

    def get_source_label(self) -> str:
        return "live_database"

    def get_quality_report(self) -> Dict[str, Any]:
        return self._quality_report

    async def get_total_jobs_by_month(self) -> Dict[str, int]:
        """Queries distinct job postings grouped by posting month."""
        if not self.db_target:
            self._quality_report["errors"].append("No database connection or session provided.")
            return {}

        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import AsyncSession

            query = text("""
                SELECT
                    TO_CHAR(posting_date, 'YYYY-MM') AS year_month,
                    COUNT(id) AS total_jobs
                FROM job_postings
                GROUP BY TO_CHAR(posting_date, 'YYYY-MM')
                ORDER BY year_month ASC
            """)

            if isinstance(self.db_target, AsyncSession):
                result = await self.db_target.execute(query)
                rows = result.fetchall()
            else:
                # If engine or sync session
                rows = []

            totals = {row[0]: int(row[1]) for row in rows if row[0]}
            self._quality_report["connected"] = True
            return totals
        except Exception as e:
            self._quality_report["errors"].append(f"Database query failed: {e}")
            return {}

    async def get_monthly_skill_demand(
        self, skills: Optional[List[str]] = None
    ) -> Dict[str, List[HistoricalObservation]]:
        """
        Dynamically aggregates monthly skill counts by joining
        job_postings -> job_skills -> skill_taxonomy.
        """
        totals_by_month = await self.get_total_jobs_by_month()
        if not totals_by_month:
            return {}

        try:
            from sqlalchemy import text
            from sqlalchemy.ext.asyncio import AsyncSession

            sql = """
                SELECT
                    TO_CHAR(p.posting_date, 'YYYY-MM') AS year_month,
                    t.canonical_name AS skill_name,
                    COUNT(DISTINCT p.id) AS skill_count
                FROM job_postings p
                JOIN job_skills js ON p.id = js.job_posting_id
                JOIN skill_taxonomy t ON js.skill_id = t.id
            """
            params = {}
            if skills:
                sql += " WHERE t.canonical_name = ANY(:skills)"
                params["skills"] = skills

            sql += """
                GROUP BY TO_CHAR(p.posting_date, 'YYYY-MM'), t.canonical_name
                ORDER BY year_month ASC
            """

            if isinstance(self.db_target, AsyncSession):
                result = await self.db_target.execute(text(sql), params)
                rows = result.fetchall()
            else:
                rows = []

            skill_obs_map: Dict[str, List[HistoricalObservation]] = defaultdict(list)
            for row in rows:
                ym, s_name, count = row[0], row[1], int(row[2])
                m_total = totals_by_month.get(ym, 0)
                share = round(count / m_total, 4) if m_total > 0 else 0.0
                skill_obs_map[s_name].append(
                    HistoricalObservation(
                        year_month=ym,
                        count=count,
                        total_jobs=m_total,
                        demand_share=share,
                    )
                )

            final_series = {}
            for s_name, obs_list in skill_obs_map.items():
                final_series[s_name] = fill_missing_months(
                    obs_list, total_jobs_by_month=totals_by_month, zero_fill=True
                )

            return final_series
        except Exception as e:
            self._quality_report["errors"].append(f"Failed to aggregate skills from database: {e}")
            return {}
