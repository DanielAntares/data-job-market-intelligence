"""Normalized job-posting schema shared by every source.

Each API returns its own idiosyncratic shape. We map all of them onto this one
``JobPosting`` record so the rest of the pipeline (storage, analysis, modeling)
never has to care where a posting came from.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


# The canonical column order for the warehouse. Keeping it explicit makes the
# parquet schema stable even when a given run produces no rows.
COLUMNS = [
    "job_id",
    "source",
    "source_id",
    "market",
    "title",
    "company",
    "location",
    "remote",
    "category",
    "tags",
    "job_type",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
    "salary_is_predicted",
    "salary_raw",
    "description_text",
    "url",
    "posted_at",
    "collected_at",
]


@dataclass
class JobPosting:
    """One normalized posting. Sources are responsible for filling this in."""

    source: str
    source_id: str
    title: str
    market: str = ""  # which job market this posting belongs to (e.g. "Indonesia")
    company: str = ""
    location: str = ""
    remote: bool = False
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    job_type: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None  # "yearly" | "monthly" | "hourly"
    # True when the salary was *estimated* by the source (e.g. Adzuna's model)
    # rather than advertised in the posting. Critical: exclude these when
    # training/evaluating a salary model, or you're modeling someone else's model.
    salary_is_predicted: bool = False
    salary_raw: str | None = None
    description_text: str = ""
    url: str = ""
    posted_at: str | None = None  # ISO date, source-reported
    collected_at: str = ""

    @property
    def job_id(self) -> str:
        """Stable, source-scoped id so the same posting dedupes across runs."""
        raw = f"{self.source}:{self.source_id}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def to_row(self) -> dict:
        if not self.collected_at:
            self.collected_at = datetime.now(timezone.utc).isoformat()
        row = asdict(self)
        row["job_id"] = self.job_id
        return {col: row.get(col) for col in COLUMNS}
