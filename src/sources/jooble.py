"""Jooble — global job aggregator that *does* cover Indonesia & Malaysia.

This is the source that makes the project relevant to the local SE-Asian market
(Adzuna has no Indonesia/Malaysia coverage). Jooble aggregates many local boards
and is driven by a free-text ``location`` rather than a country code.

Auth: a free API key from https://jooble.org/api/about — request it, then put it
in ``.env`` as ``JOOBLE_API_KEY``.

API: POST https://jooble.org/api/<key>  with JSON ``{"keywords", "location"}``.
Salary arrives as free text (often empty / local currency), so coverage of
*advertised* pay is thin — Jooble's value here is local demand signal.
Docs: https://jooble.org/api/about
"""
from __future__ import annotations

import os

from ..salary import parse_salary
from ..schema import JobPosting
from .base import BaseSource, clean_field, html_to_text, is_data_role

_ENDPOINT = "https://jooble.org/api/"

# Fallback currency by market, used when the salary text has an amount but no
# recognisable currency symbol (Jooble's "$" handling is parsed in salary.py).
_MARKET_CURRENCY = {
    "indonesia": "IDR", "malaysia": "MYR", "australia": "AUD",
    "singapore": "SGD", "philippines": "PHP", "thailand": "THB",
}


class JoobleSource(BaseSource):
    name = "jooble"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.api_key = os.getenv("JOOBLE_API_KEY")
        self.locations = self.config.get("locations") or ["Indonesia"]

    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        if not self.api_key:
            raise RuntimeError(
                "Jooble enabled but JOOBLE_API_KEY not set (add it to .env). "
                "Get a free key at https://jooble.org/api/about."
            )
        seen: dict[str, dict] = {}
        per_term = max(1, limit // max(1, len(search_terms) * len(self.locations)))
        for location in self.locations:
            for term in search_terms:
                payload = self._post_json(
                    _ENDPOINT + self.api_key,
                    {"keywords": term, "location": location},
                )
                for job in (payload.get("jobs") or [])[:per_term]:
                    job["_market"] = location
                    key = str(job.get("id") or job.get("link"))
                    seen[f"{location}:{key}"] = job
        return list(seen.values())

    def normalize(self, record: dict) -> JobPosting | None:
        title = record.get("title")
        # Jooble's keyword search is loose (returns many non-data roles), so
        # keep only postings whose title is genuinely a data role.
        if not is_data_role(title):
            return None
        market = record.get("_market", "")
        sal = parse_salary(record.get("salary"))
        # If we parsed an amount but no currency symbol, fall back to the market's.
        if sal["salary_min"] is not None and not sal["salary_currency"]:
            sal["salary_currency"] = _MARKET_CURRENCY.get(market.lower())

        key = str(record.get("id") or record.get("link"))
        loc_text = record.get("location") or ""
        return JobPosting(
            source=self.name,
            source_id=f"{market}:{key}",
            title=clean_field(title),
            market=market,
            company=clean_field(record.get("company")),
            location=clean_field(loc_text),
            remote="remote" in f"{title} {loc_text}".lower(),
            category=None,  # Jooble has no category taxonomy
            tags=[],
            job_type=record.get("type") or None,
            salary_raw=record.get("salary") or None,
            description_text=html_to_text(record.get("snippet")),
            url=record.get("link", ""),
            posted_at=(record.get("updated") or "")[:10] or None,
            **sal,
        )
