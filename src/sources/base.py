"""Common base class for every job source."""
from __future__ import annotations

import html
import re
from abc import ABC, abstractmethod

import requests
from bs4 import BeautifulSoup

from ..schema import JobPosting

_USER_AGENT = "data-job-market-intelligence/0.1 (portfolio project)"


# Real data-role title patterns. Used to filter sources whose keyword search is
# loose (Jooble) or whose tags are spammy (RemoteOK) down to actual data roles.
DATA_ROLE_TITLE_PATTERNS = (
    "data analyst", "data scientist", "data engineer", "data science",
    "machine learning", "analytics engineer", "ml engineer", "mlops",
    "business intelligence", "bi developer", "bi analyst", "analytics manager",
    "head of data", "data lead", "ai engineer", "data architect",
    "analytics specialist", "data specialist",
)


def is_data_role(title: str | None) -> bool:
    """True if a job title looks like a genuine data role (not data-entry)."""
    t = (title or "").lower()
    if "data entry" in t:
        return False
    return any(p in t for p in DATA_ROLE_TITLE_PATTERNS)


def clean_field(value: str | None) -> str:
    """Tidy a short plain-text field: decode HTML entities, collapse spaces.

    Source feeds sneak entities into 'plain' fields (e.g. a category of
    'Data Science &amp; Analytics', or a company 'Smith &amp; Co'), so we
    decode them here rather than letting '&amp;' leak into analysis.
    """
    if not value:
        return ""
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def html_to_text(raw_html: str | None) -> str:
    """Strip HTML to readable plain text for downstream NLP."""
    if not raw_html:
        return ""
    text = BeautifulSoup(raw_html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


class BaseSource(ABC):
    """A job board we can pull postings from.

    Subclasses implement two things:
      * ``fetch_raw``  — hit the API, return the list of raw record dicts.
      * ``normalize``  — map one raw record onto a ``JobPosting``.
    """

    name: str = "base"

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": _USER_AGENT})

    def _get_json(self, url: str, params: dict | None = None) -> dict | list:
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _post_json(self, url: str, payload: dict) -> dict | list:
        resp = self.session.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @abstractmethod
    def fetch_raw(self, search_terms: list[str], limit: int) -> list[dict]:
        ...

    @abstractmethod
    def normalize(self, record: dict) -> JobPosting | None:
        ...

    def collect(self, search_terms: list[str], limit: int) -> tuple[list[dict], list[JobPosting]]:
        """Fetch + normalize. Returns (raw_records, postings)."""
        raw = self.fetch_raw(search_terms, limit)
        postings: list[JobPosting] = []
        for rec in raw:
            try:
                posting = self.normalize(rec)
            except Exception:
                posting = None
            if posting is not None:
                postings.append(posting)
        return raw, postings
