"""
CSV Data Source Adapter: Ingests and validates relational CSV exports or pre-aggregated CSVs.
Supports dynamic aggregation across job_postings, job_skills, and skill_taxonomy.
"""

from collections import defaultdict
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from services.ai_forecasting.data.interfaces import DataSourceAdapter
from services.ai_forecasting.data.aggregator import (
    fill_missing_months,
    validate_monthly_observations,
)
from services.ai_forecasting.schemas import HistoricalObservation


class CSVDataSource(DataSourceAdapter):
    """
    Adapter that loads CSV datasets.
    Supports either:
    1. Relational trio: job_postings.csv + job_skills.csv + skill_taxonomy.csv
    2. Pre-aggregated monthly skill demand CSV (year_month, skill_name, count, total_jobs)
    """

    def __init__(
        self,
        fixtures_dir: Optional[Path] = None,
        pre_aggregated_path: Optional[Path] = None,
    ):
        self.fixtures_dir = fixtures_dir
        self.pre_aggregated_path = pre_aggregated_path
        self._quality_report: Dict[str, Any] = {
            "source": "csv",
            "warnings": [],
            "errors": [],
            "postings_count": 0,
            "skills_count": 0,
            "links_count": 0,
            "distinct_months": [],
        }

    def get_source_label(self) -> str:
        return "csv"

    def get_quality_report(self) -> Dict[str, Any]:
        return self._quality_report

    def _load_pre_aggregated(self, filepath: Path) -> Tuple[Dict[str, List[HistoricalObservation]], Dict[str, int]]:
        if not filepath.exists():
            raise FileNotFoundError(f"Pre-aggregated CSV not found at: {filepath}")

        total_jobs_by_month: Dict[str, int] = {}
        skill_obs_map: Dict[str, List[HistoricalObservation]] = defaultdict(list)

        with open(filepath, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required_cols = {"year_month", "skill_name", "count", "total_jobs"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"Pre-aggregated CSV missing required columns. Expected {required_cols}, got {reader.fieldnames}")

            for row_idx, row in enumerate(reader, start=2):
                ym = row["year_month"].strip()
                s_name = row["skill_name"].strip()
                try:
                    count = int(row["count"].strip())
                    total_jobs = int(row["total_jobs"].strip())
                except ValueError as e:
                    raise ValueError(f"Row {row_idx}: count and total_jobs must be integers. Details: {e}")

                if count < 0:
                    self._quality_report["warnings"].append(f"Row {row_idx}: Negative count {count} for {s_name} in {ym}")
                    count = 0
                if total_jobs < 0:
                    total_jobs = 0

                # Track total jobs per month
                if ym not in total_jobs_by_month or total_jobs > total_jobs_by_month[ym]:
                    total_jobs_by_month[ym] = total_jobs

                share = round(count / total_jobs, 4) if total_jobs > 0 else 0.0
                obs = HistoricalObservation(
                    year_month=ym,
                    count=count,
                    total_jobs=total_jobs,
                    demand_share=share,
                )
                skill_obs_map[s_name].append(obs)

        # Sort and fill missing months
        final_series: Dict[str, List[HistoricalObservation]] = {}
        for s_name, obs_list in skill_obs_map.items():
            obs_list.sort(key=lambda x: x.year_month)
            filled = fill_missing_months(obs_list, total_jobs_by_month=total_jobs_by_month, zero_fill=True)
            final_series[s_name] = filled

        self._quality_report["distinct_months"] = sorted(total_jobs_by_month.keys())
        self._quality_report["skills_count"] = len(final_series)
        return final_series, total_jobs_by_month

    def _load_relational_trio(self, fixtures_dir: Path) -> Tuple[Dict[str, List[HistoricalObservation]], Dict[str, int]]:
        postings_file = fixtures_dir / "job_postings.csv"
        skills_file = fixtures_dir / "job_skills.csv"
        taxonomy_file = fixtures_dir / "skill_taxonomy.csv"

        for p in [postings_file, skills_file, taxonomy_file]:
            if not p.exists():
                raise FileNotFoundError(f"Required relational CSV not found: {p}")

        # 1. Ingest job_postings: id -> (job_id, posting_date)
        postings: Dict[int, str] = {}  # id -> YYYY-MM
        all_posting_dates: Set[str] = set()

        with open(postings_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"id", "job_id", "job_hash", "posting_date"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"job_postings.csv missing required columns: {required}")

            for row in reader:
                try:
                    p_id = int(row["id"].strip())
                    p_date_str = row["posting_date"].strip()
                    # Validate date
                    p_date = datetime.strptime(p_date_str, "%Y-%m-%d").date()
                    ym = f"{p_date.year:04d}-{p_date.month:02d}"
                    postings[p_id] = ym
                    all_posting_dates.add(p_date_str)
                except Exception as e:
                    self._quality_report["warnings"].append(f"Invalid row in job_postings: {row}. Details: {e}")

        self._quality_report["postings_count"] = len(postings)

        # Calculate total_jobs per month independently from distinct job postings!
        total_jobs_by_month: Dict[str, int] = defaultdict(int)
        for ym in postings.values():
            total_jobs_by_month[ym] += 1

        # 2. Ingest skill_taxonomy: id -> canonical_name
        taxonomy: Dict[int, str] = {}
        with open(taxonomy_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"id", "canonical_name"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"skill_taxonomy.csv missing required columns: {required}")

            for row in reader:
                try:
                    s_id = int(row["id"].strip())
                    c_name = row["canonical_name"].strip()
                    if c_name:
                        taxonomy[s_id] = c_name
                except Exception as e:
                    self._quality_report["warnings"].append(f"Invalid row in skill_taxonomy: {row}. Details: {e}")

        self._quality_report["skills_count"] = len(taxonomy)

        # 3. Ingest job_skills: (job_posting_id, skill_id)
        # Using a set to guarantee distinct (job_posting_id, skill_id)
        linked_postings: Set[int] = set()
        skill_monthly_postings: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
        duplicate_links = 0

        with open(skills_file, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"id", "job_posting_id", "skill_id"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"job_skills.csv missing required columns: {required}")

            for row in reader:
                try:
                    p_id = int(row["job_posting_id"].strip())
                    s_id = int(row["skill_id"].strip())

                    # Foreign key checks
                    if p_id not in postings:
                        self._quality_report["warnings"].append(
                            f"job_skills row references unknown job_posting_id {p_id}"
                        )
                        continue
                    if s_id not in taxonomy:
                        self._quality_report["warnings"].append(
                            f"job_skills row references unknown skill_id {s_id}"
                        )
                        continue

                    canonical_skill = taxonomy[s_id]
                    ym = postings[p_id]

                    if p_id in skill_monthly_postings[canonical_skill][ym]:
                        duplicate_links += 1
                    else:
                        skill_monthly_postings[canonical_skill][ym].add(p_id)
                    linked_postings.add(p_id)

                except Exception as e:
                    self._quality_report["warnings"].append(f"Invalid row in job_skills: {row}. Details: {e}")

        self._quality_report["links_count"] = len(linked_postings)
        if duplicate_links > 0:
            self._quality_report["warnings"].append(f"Ignored {duplicate_links} duplicate (job_posting_id, skill_id) links.")

        unlinked_postings_count = len(postings) - len(linked_postings)
        if unlinked_postings_count > 0:
            self._quality_report["warnings"].append(
                f"{unlinked_postings_count} of {len(postings)} job postings have no skills linked."
            )

        distinct_months = sorted(total_jobs_by_month.keys())
        self._quality_report["distinct_months"] = distinct_months

        if len(distinct_months) < 4:
            self._quality_report["warnings"].append(
                f"Historical timespan is too short: {len(distinct_months)} month(s) available ({distinct_months}). "
                "Minimum 4 months required for time-series forecasting. Insufficient data."
            )

        # Assemble HistoricalObservation per skill
        final_series: Dict[str, List[HistoricalObservation]] = {}
        for s_id, s_name in taxonomy.items():
            obs_list = []
            for ym in distinct_months:
                count = len(skill_monthly_postings[s_name].get(ym, set()))
                tot = total_jobs_by_month[ym]
                share = round(count / tot, 4) if tot > 0 else 0.0
                obs_list.append(
                    HistoricalObservation(
                        year_month=ym,
                        count=count,
                        total_jobs=tot,
                        demand_share=share,
                    )
                )
            final_series[s_name] = obs_list

        return final_series, total_jobs_by_month

    async def get_monthly_skill_demand(
        self, skills: Optional[List[str]] = None
    ) -> Dict[str, List[HistoricalObservation]]:
        if self.pre_aggregated_path and self.pre_aggregated_path.exists():
            series, _ = self._load_pre_aggregated(self.pre_aggregated_path)
        elif self.fixtures_dir and self.fixtures_dir.exists():
            series, _ = self._load_relational_trio(self.fixtures_dir)
        else:
            raise FileNotFoundError("No valid CSV directory or pre-aggregated path configured.")

        if skills is None:
            return series

        return {k: v for k, v in series.items() if k in skills}

    async def get_total_jobs_by_month(self) -> Dict[str, int]:
        if self.pre_aggregated_path and self.pre_aggregated_path.exists():
            _, totals = self._load_pre_aggregated(self.pre_aggregated_path)
            return totals
        elif self.fixtures_dir and self.fixtures_dir.exists():
            _, totals = self._load_relational_trio(self.fixtures_dir)
            return totals
        return {}
