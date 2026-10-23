"""
Deterministic synthetic monthly skill-demand data generator.
Produces labeled demo datasets for testing and demonstrations without requiring a live database.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
from services.ai_forecasting.data.interfaces import DataSourceAdapter
from services.ai_forecasting.data.aggregator import add_months, generate_month_range
from services.ai_forecasting.schemas import HistoricalObservation


class SyntheticDemoDataSource(DataSourceAdapter):
    """
    Generates deterministic, mathematically modeled monthly skill demand curves.
    Clearly isolated and labeled as 'synthetic_demo'.
    """

    def __init__(
        self,
        reference_month: Optional[str] = None,
        months_count: int = 24,
        start_year: Optional[int] = None,
        seed: int = 42,
    ):
        self.reference_month = reference_month
        self.start_year = start_year
        self.months_count = months_count
        self.seed = seed
        self._cache: Optional[Dict[str, List[HistoricalObservation]]] = None
        self._total_jobs_cache: Optional[Dict[str, int]] = None

    def get_source_label(self) -> str:
        return "synthetic_demo"

    def get_quality_report(self) -> Dict[str, any]:
        last_m = self.reference_month or datetime.now(timezone.utc).strftime("%Y-%m")
        if self.start_year is not None and self.reference_month is None:
            last_m = add_months(f"{self.start_year:04d}-01", self.months_count - 1)
        return {
            "source": "synthetic_demo",
            "is_synthetic": True,
            "seed": self.seed,
            "months_count": self.months_count,
            "last_historical_month": last_m,
            "status": "deterministic_demo_ready",
            "notes": "Synthetic benchmark curves generated for testing/demo purposes. Not actual market data.",
        }

    def _generate(self) -> Tuple[Dict[str, List[HistoricalObservation]], Dict[str, int]]:
        rng = np.random.RandomState(self.seed)
        months: List[str] = []
        total_jobs_by_month: Dict[str, int] = {}

        # If start_year is explicitly provided and reference_month is None, preserve explicit start year
        if self.start_year is not None and self.reference_month is None:
            current_year = self.start_year
            current_month = 1
            for _ in range(self.months_count):
                ym = f"{current_year:04d}-{current_month:02d}"
                months.append(ym)
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
        else:
            # Anchor to reference_month (defaulting to current execution month YYYY-MM)
            ref_ym = self.reference_month or datetime.now(timezone.utc).strftime("%Y-%m")
            start_ym = add_months(ref_ym, -(self.months_count - 1))
            months = generate_month_range(start_ym, ref_ym)

        for idx, ym in enumerate(months):
            base_total = 400 + int(idx * 4.5) + rng.randint(-10, 10)
            total_jobs_by_month[ym] = max(100, base_total)

        skill_series: Dict[str, List[HistoricalObservation]] = {
            "Python": [],
            "FastAPI": [],
            "jQuery": [],
            "Rust": [],
        }

        for idx, ym in enumerate(months):
            tot = total_jobs_by_month[ym]
            t = idx / (self.months_count - 1)  # 0.0 to 1.0

            # 1. Python (Core / Stable): ~125 - 155 with slight seasonal cycle and subtle baseline trend
            py_count = int(128 + 5.0 * t + 15 * np.sin(2 * np.pi * (idx % 12) / 12) + rng.randint(-4, 5))
            py_share = round(py_count / tot, 4)
            skill_series["Python"].append(
                HistoricalObservation(year_month=ym, count=py_count, total_jobs=tot, demand_share=py_share)
            )

            # 2. FastAPI (Emerging): rapid upward growth from ~20 to ~125
            fa_count = int(22 + 105 * (t ** 1.3) + rng.randint(-3, 4))
            fa_share = round(fa_count / tot, 4)
            skill_series["FastAPI"].append(
                HistoricalObservation(year_month=ym, count=fa_count, total_jobs=tot, demand_share=fa_share)
            )

            # 3. jQuery (Declining): persistent drop from ~90 down to ~22
            jq_count = max(5, int(92 - 70 * (t ** 0.9) + rng.randint(-3, 4)))
            jq_share = round(jq_count / tot, 4)
            skill_series["jQuery"].append(
                HistoricalObservation(year_month=ym, count=jq_count, total_jobs=tot, demand_share=jq_share)
            )

            # 4. Rust (Niche): steady low volume (~8 to 16), demand_share < 0.05
            ru_count = int(8 + 8 * t + rng.randint(-2, 3))
            ru_share = round(ru_count / tot, 4)
            skill_series["Rust"].append(
                HistoricalObservation(year_month=ym, count=ru_count, total_jobs=tot, demand_share=ru_share)
            )

        return skill_series, total_jobs_by_month

    async def get_monthly_skill_demand(
        self, skills: Optional[List[str]] = None
    ) -> Dict[str, List[HistoricalObservation]]:
        if self._cache is None or self._total_jobs_cache is None:
            self._cache, self._total_jobs_cache = self._generate()

        if skills is None:
            return self._cache

        filtered = {}
        for s in skills:
            if s in self._cache:
                filtered[s] = self._cache[s]
            else:
                # If an unrecognized skill is requested in demo mode, generate a moderate baseline
                rng = np.random.RandomState(abs(hash(s)) % (2**31 - 1))
                demo_obs = []
                for ym, tot in self._total_jobs_cache.items():
                    c = int(25 + rng.randint(-5, 6))
                    demo_obs.append(
                        HistoricalObservation(year_month=ym, count=c, total_jobs=tot, demand_share=round(c / tot, 4))
                    )
                filtered[s] = demo_obs
        return filtered

    async def get_total_jobs_by_month(self) -> Dict[str, int]:
        if self._total_jobs_cache is None:
            self._cache, self._total_jobs_cache = self._generate()
        return self._total_jobs_cache
