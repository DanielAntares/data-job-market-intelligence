"""Adzuna — job-search aggregator with broad, non-remote coverage.

Why it's here: Remotive and Jobicy are remote-only, which biases the dataset.
Adzuna aggregates employers and boards across whole countries (on-site, hybrid,
remote) with structured salaries and geography — correcting that bias and adding
the volume a salary model needs.

Auth: free ``app_id`` / ``app_key`` from https://developer.adzuna.com/, read
from the environment (loaded from a git-ignored ``.env``).

Important: Adzuna sets ``salary_is_predicted = "1"`` when *it* estimated the pay
rather than the posting advertising it. We carry that flag through so downstream
analysis/modeling can exclude estimates.
Docs: https://developer.adzuna.com/docs/search
"""
from __future__ import annotations

import os

from ..schema import JobPosting
from .base import BaseSource, clean_field, html_to_text

_BASE = "https://api.adzuna.com/v1/api/jobs"
_MAX_PER_PAGE = 50  # Adzuna's per-request cap

# Default advertised currency by country (Adzuna salaries are annual).
_COUNTRY_CURRENCY = {
    "us": "USD", "gb": "GBP", "ca": "CAD", "au": "AUD", "in": "INR",
    "de": "EUR", "fr": "EUR", "es": "EUR", "it": "EUR", "nl": "EUR",
}

# Human-readable market label per country (note: Adzuna has no Indonesia/Malaysia
# support — those come from Jooble instead).
_COUNTRY_MARKET = {
    "us": "United States", "au": "Australia", "gb": "United Kingdom",
    "ca": "Canada", "in": "India", "sg": "Singapore", "nz": "New Zealand",
}


class AdzunaSource(BaseSource):
    name = "adzuna"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # Accept either `countries: [us, au]` or a single legacy `country: us`.
        countries = self.config.get("countries") or [self.config.get("country") or "us"]
        self.countries = [c.lower() for c in countries]
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")

    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        if not self.app_id or not self.app_key:
            raise RuntimeError(
                "Adzuna enabled but ADZUNA_APP_ID / ADZUNA_APP_KEY not set "
                "(add them to .env). See README 'Configuring sources'."
            )
        seen: dict[str, dict] = {}
        per_term = max(1, limit // max(1, len(search_terms) * len(self.countries)))
        for country in self.countries:
            for term in search_terms:
                collected, page = 0, 1
                while collected < per_term:
                    want = min(_MAX_PER_PAGE, per_term - collected)
                    payload = self._get_json(
                        f"{_BASE}/{country}/search/{page}",
                        params={
                            "app_id": self.app_id,
                            "app_key": self.app_key,
                            "results_per_page": want,
                            "what": term,
                            "content-type": "application/json",
                        },
                    )
                    results = payload.get("results", [])
                    if not results:
                        break  # no more pages for this term
                    for job in results:
                        job["_country"] = country  # tag so normalize knows the market
                        seen[f"{country}:{job['id']}"] = job
                    collected += len(results)
                    page += 1
        return list(seen.values())

    def normalize(self, record: dict) -> JobPosting | None:
        title = record.get("title")
        if not title:
            return None
        country = record.get("_country", self.countries[0])
        title = clean_field(title)
        company = clean_field((record.get("company") or {}).get("display_name"))
        location = clean_field((record.get("location") or {}).get("display_name"))
        category = clean_field((record.get("category") or {}).get("label")) or None

        # Adzuna has no explicit remote flag; infer it from the text.
        remote = "remote" in f"{title} {location}".lower()

        # contract_time -> job_type (full_time / part_time), when present.
        job_type = record.get("contract_time") or record.get("contract_type")

        smin = record.get("salary_min")
        smax = record.get("salary_max")
        predicted = str(record.get("salary_is_predicted", "0")) == "1"
        created = (record.get("created") or "")[:10] or None

        return JobPosting(
            source=self.name,
            source_id=f"{country}:{record['id']}",
            title=title,
            market=_COUNTRY_MARKET.get(country, country.upper()),
            company=company,
            location=location,
            remote=remote,
            category=category,
            tags=[],  # Adzuna doesn't expose skill tags
            job_type=job_type,
            salary_min=float(smin) if smin is not None else None,
            salary_max=float(smax) if smax is not None else None,
            salary_currency=_COUNTRY_CURRENCY.get(country) if (smin or smax) else None,
            salary_period="yearly" if (smin or smax) else None,
            salary_is_predicted=predicted,
            salary_raw=None,
            description_text=html_to_text(record.get("description")),
            url=record.get("redirect_url", ""),
            posted_at=created,
        )
