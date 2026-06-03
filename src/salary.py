"""Parse messy free-text salary strings into structured numbers.

Sources like Remotive report salary as human text: ``"$80k - $100k"``,
``"$18 - $22/hr"``, ``"$90 - $150 /hour"``, ``"€50k per year"``. Turning that
into ``(min, max, currency, period)`` is exactly the kind of real-world
data-wrangling a clean Kaggle CSV never makes you do — so it gets its own
module and its own unit tests (see ``tests/test_salary.py``).
"""
from __future__ import annotations

import re

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP"}
_CURRENCY_CODES = ("USD", "EUR", "GBP", "CAD", "AUD", "INR")

# A number, optionally with thousands commas, a decimal, and a 'k' multiplier.
_NUMBER_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*([kK])?")


def _detect_currency(text: str) -> str | None:
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    upper = text.upper()
    for code in _CURRENCY_CODES:
        if code in upper:
            return code
    return None


def _detect_period(text: str) -> str | None:
    t = text.lower()
    if re.search(r"\b(hour|hourly|/\s*hr|per\s*hr|/\s*hour|hr)\b", t) or "/hr" in t:
        return "hourly"
    if re.search(r"\b(month|monthly|/\s*mo|per\s*month|/mo)\b", t):
        return "monthly"
    if re.search(r"\b(year|yearly|annual|annually|annum|per\s*year|/\s*yr|/yr|pa)\b", t):
        return "yearly"
    return None


def _to_number(digits: str, k_suffix: str | None) -> float:
    value = float(digits.replace(",", ""))
    if k_suffix:
        value *= 1000
    return value


def parse_salary(text: str | None) -> dict:
    """Return ``{salary_min, salary_max, salary_currency, salary_period}``.

    Missing pieces come back as ``None`` rather than guesses we can't justify.
    """
    blank = {
        "salary_min": None,
        "salary_max": None,
        "salary_currency": None,
        "salary_period": None,
    }
    if not text or not isinstance(text, str):
        return blank

    matches = _NUMBER_RE.findall(text)
    numbers = [_to_number(d, k) for d, k in matches if d.strip(",")]
    if not numbers:
        return blank

    currency = _detect_currency(text)
    period = _detect_period(text)

    low = min(numbers)
    high = max(numbers) if len(numbers) > 1 else None

    # Heuristic for shorthand ranges like "$80-100k", where only the upper
    # bound carries the 'k': if the low looks un-scaled next to a thousands-
    # scale high, lift it to the same magnitude.
    if high is not None and low < 1000 <= high and period != "hourly":
        low *= 1000

    # If no explicit period, infer from magnitude: small numbers are hourly
    # rates, large numbers are annual.
    if period is None:
        reference = high if high is not None else low
        period = "hourly" if reference < 1000 else "yearly"

    return {
        "salary_min": low,
        "salary_max": high,
        "salary_currency": currency,
        "salary_period": period,
    }
