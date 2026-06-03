"""RemoteOK — popular remote-jobs board, free no-auth JSON API.

Real-world catch handled here: RemoteOK applies **SEO-spam tags** (unrelated
listings tagged `data` / `machine learning` to farm views), so filtering on tags
pulls in admin/VA/data-entry junk. We instead pull a few data-relevant tag feeds
for breadth, then keep only postings whose **title** matches a real data role.

Salaries are structured USD (`salary_min`/`salary_max`), though often 0 (= not
disclosed). Market: remote-international.
Docs: https://remoteok.com/api
"""
from __future__ import annotations

from ..schema import JobPosting
from .base import BaseSource, clean_field, html_to_text, is_data_role

_BASE = "https://remoteok.com/api"
_TAG_FEEDS = ("data", "data-science", "machine-learning", "analytics")


class RemoteOKSource(BaseSource):
    name = "remoteok"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # RemoteOK blocks non-browser user agents.
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; portfolio-project)"})

    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        seen: dict[str, dict] = {}
        for tag in _TAG_FEEDS:
            payload = self._get_json(_BASE, params={"tags": tag})
            for job in payload:
                if isinstance(job, dict) and job.get("position"):
                    seen[str(job["id"])] = job
        return list(seen.values())

    def normalize(self, record: dict) -> JobPosting | None:
        title = record.get("position")
        if not is_data_role(title):
            return None
        smin = record.get("salary_min") or 0
        smax = record.get("salary_max") or 0
        has_salary = smin and smin > 0
        return JobPosting(
            source=self.name,
            source_id=str(record.get("id")),
            title=clean_field(title),
            market="Remote (international)",
            company=clean_field(record.get("company")),
            location=clean_field(record.get("location")),
            remote=True,
            category=None,
            tags=record.get("tags") or [],
            job_type=None,
            salary_min=float(smin) if has_salary else None,
            salary_max=float(smax) if (smax and smax > 0) else None,
            salary_currency="USD" if has_salary else None,
            salary_period="yearly" if has_salary else None,
            salary_raw=None,
            description_text=html_to_text(record.get("description")),
            url=record.get("url") or record.get("apply_url", ""),
            posted_at=(record.get("date") or "")[:10] or None,
        )
