"""Jobicy — free, no-auth remote-jobs API.

Notable trait: salary is already structured (``salaryMin``/``salaryMax``/
``salaryCurrency``/``salaryPeriod``), giving us clean numeric targets for the
salary model later — a nice complement to Remotive's messy text.
Docs: https://jobicy.com/jobs-rss-feed
"""
from __future__ import annotations

from ..schema import JobPosting
from .base import BaseSource, clean_field, html_to_text

_ENDPOINT = "https://jobicy.com/api/v2/remote-jobs"
_MAX_COUNT = 50  # Jobicy caps `count` at 50 per request


def _first(value):
    """Jobicy returns some fields as single-item lists; flatten them."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


class JobicySource(BaseSource):
    name = "jobicy"

    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        seen: dict[str, dict] = {}
        count = min(limit, _MAX_COUNT)
        for term in search_terms:
            # Jobicy's `tag` is a keyword match; single keywords work best, so
            # we send the most specific word of each term.
            tag = term.split()[-1]
            payload = self._get_json(_ENDPOINT, params={"count": count, "tag": tag})
            for job in payload.get("jobs", []):
                seen[str(job["id"])] = job
        return list(seen.values())

    def normalize(self, record: dict) -> JobPosting | None:
        if not record.get("jobTitle"):
            return None
        period = (record.get("salaryPeriod") or "").lower() or None
        posted = (record.get("pubDate") or "")[:10] or None
        return JobPosting(
            source=self.name,
            source_id=str(record["id"]),
            title=clean_field(record["jobTitle"]),
            market="Remote (international)",
            company=clean_field(record.get("companyName")),
            location=clean_field(record.get("jobGeo")),
            remote=True,
            category=clean_field(_first(record.get("jobIndustry"))) or None,
            tags=record.get("jobIndustry") if isinstance(record.get("jobIndustry"), list) else [],
            job_type=_first(record.get("jobType")),
            salary_min=record.get("salaryMin") or None,
            salary_max=record.get("salaryMax") or None,
            salary_currency=record.get("salaryCurrency") or None,
            salary_period=period,
            salary_raw=None,
            description_text=html_to_text(record.get("jobDescription") or record.get("jobExcerpt")),
            url=record.get("url", ""),
            posted_at=posted,
        )
