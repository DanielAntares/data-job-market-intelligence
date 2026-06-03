"""Remotive — free, no-auth remote-jobs API.

Notable trait: salary arrives as free text (``"$80k - $100k"``, ``"$18/hr"``),
so we lean on ``salary.parse_salary`` to structure it.
Docs: https://remotive.com/api/remote-jobs
"""
from __future__ import annotations

from ..salary import parse_salary
from ..schema import JobPosting
from .base import BaseSource, clean_field, html_to_text

_ENDPOINT = "https://remotive.com/api/remote-jobs"


class RemotiveSource(BaseSource):
    name = "remotive"

    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        seen: dict[int, dict] = {}
        per_term = max(1, limit // max(1, len(search_terms)))
        for term in search_terms:
            payload = self._get_json(
                _ENDPOINT, params={"search": term, "limit": per_term}
            )
            for job in payload.get("jobs", []):
                seen[job["id"]] = job  # dedupe across terms by Remotive id
        return list(seen.values())

    def normalize(self, record: dict) -> JobPosting | None:
        if not record.get("title"):
            return None
        sal = parse_salary(record.get("salary"))
        posted = (record.get("publication_date") or "")[:10] or None
        return JobPosting(
            source=self.name,
            source_id=str(record["id"]),
            title=clean_field(record["title"]),
            market="Remote (international)",
            company=clean_field(record.get("company_name")),
            location=clean_field(record.get("candidate_required_location")),
            remote=True,  # Remotive is a remote-only board
            category=clean_field(record.get("category")) or None,
            tags=record.get("tags") or [],
            job_type=record.get("job_type"),
            salary_raw=record.get("salary") or None,
            description_text=html_to_text(record.get("description")),
            url=record.get("url", ""),
            posted_at=posted,
            **sal,
        )
