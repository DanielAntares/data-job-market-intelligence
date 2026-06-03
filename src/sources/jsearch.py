"""JSearch (via RapidAPI) — aggregates Google for Jobs.

This is the closest you get *legitimately* to the big boards: Google for Jobs
pulls from LinkedIn, Indeed, Glassdoor, ZipRecruiter and more, and JSearch
exposes it through one API — globally, including Indonesia. So a single source
can serve both the international and the local market.

Auth: a free RapidAPI key (subscribe to the free JSearch tier), put it in
``.env`` as ``RAPIDAPI_KEY``. The free tier is rate-limited (a few hundred
requests/month), so keep ``num_pages`` small and run it periodically.

Salaries come from the postings (structured min/max/currency/period) when
available.
Docs: https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
"""
from __future__ import annotations

import os

from ..schema import JobPosting
from .base import BaseSource, clean_field, html_to_text

_ENDPOINT = "https://jsearch.p.rapidapi.com/search"
_HOST = "jsearch.p.rapidapi.com"
_COUNTRY_MARKET = {
    "id": "Indonesia", "my": "Malaysia", "us": "United States",
    "au": "Australia", "sg": "Singapore", "gb": "United Kingdom",
}
_PERIOD = {"YEAR": "yearly", "MONTH": "monthly", "WEEK": "weekly", "HOUR": "hourly"}


class JSearchSource(BaseSource):
    name = "jsearch"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.api_key = os.getenv("RAPIDAPI_KEY")
        self.countries = [c.lower() for c in (self.config.get("countries") or ["us"])]
        self.num_pages = int(self.config.get("num_pages", 1))
        self.work_from_home = bool(self.config.get("work_from_home", False))

    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        if not self.api_key:
            raise RuntimeError(
                "JSearch enabled but RAPIDAPI_KEY not set (add it to .env). Get a "
                "free key at https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch"
            )
        headers = {"X-RapidAPI-Key": self.api_key, "X-RapidAPI-Host": _HOST}
        seen: dict[str, dict] = {}
        for country in self.countries:
            # JSearch needs the country *name* in the query text (not just the
            # country code) to return local results — e.g. "data analyst
            # Indonesia" + country=id works; bare country=id returns nothing.
            market_name = _COUNTRY_MARKET.get(country, country)
            for term in search_terms:
                params = {
                    "query": f"{term} {market_name}", "page": "1",
                    "num_pages": str(self.num_pages),
                    "country": country, "date_posted": "all",
                }
                if self.work_from_home:
                    params["work_from_home"] = "true"
                resp = self.session.get(_ENDPOINT, params=params, headers=headers, timeout=30)
                resp.raise_for_status()
                for job in (resp.json().get("data") or []):
                    job["_country"] = country
                    seen[str(job.get("job_id"))] = job
        return list(seen.values())

    def normalize(self, record: dict) -> JobPosting | None:
        title = record.get("job_title")
        if not title:
            return None
        country = record.get("_country", "")
        smin = record.get("job_min_salary")
        smax = record.get("job_max_salary")
        has_salary = bool(smin or smax)
        loc = ", ".join(x for x in [record.get("job_city"), record.get("job_country")] if x)
        return JobPosting(
            source=self.name,
            source_id=str(record.get("job_id")),
            title=clean_field(title),
            market=_COUNTRY_MARKET.get(country, country.upper()),
            company=clean_field(record.get("employer_name")),
            location=clean_field(loc),
            remote=bool(record.get("job_is_remote")),
            category=None,
            tags=[],
            job_type=record.get("job_employment_type"),
            salary_min=float(smin) if smin else None,
            salary_max=float(smax) if smax else None,
            salary_currency=record.get("job_salary_currency") if has_salary else None,
            salary_period=_PERIOD.get((record.get("job_salary_period") or "").upper())
            if has_salary else None,
            salary_raw=None,
            description_text=html_to_text(record.get("job_description")),
            url=record.get("job_apply_link", ""),
            posted_at=(record.get("job_posted_at_datetime_utc") or "")[:10] or None,
        )
