"""Tests for Adzuna normalization (nested fields + predicted-salary flag)."""
from src.sources.adzuna import AdzunaSource

# A trimmed real-shaped Adzuna search result.
SAMPLE = {
    "id": "456",
    "title": "Sr. Data Analyst",
    "company": {"display_name": "TEKsystems"},
    "location": {"display_name": "Chicago, Cook County", "area": ["US", "Illinois", "Chicago"]},
    "category": {"label": "IT Jobs", "tag": "it-jobs"},
    "salary_min": 97393.99,
    "salary_max": 97393.99,
    "salary_is_predicted": "1",
    "created": "2026-05-30T07:04:03Z",
    "description": "We need a <b>data analyst</b> with SQL &amp; Python.",
    "redirect_url": "https://example.com/job/456",
}


def _norm(rec):
    return AdzunaSource({"country": "us"}).normalize(rec)


def test_flattens_nested_fields():
    p = _norm(SAMPLE)
    assert p.company == "TEKsystems"
    assert p.location == "Chicago, Cook County"
    assert p.category == "IT Jobs"


def test_predicted_salary_flag_is_captured():
    p = _norm(SAMPLE)
    assert p.salary_is_predicted is True
    assert p.salary_min == 97393.99
    assert p.salary_currency == "USD"
    assert p.salary_period == "yearly"


def test_advertised_salary_not_flagged():
    rec = dict(SAMPLE, salary_is_predicted="0")
    assert _norm(rec).salary_is_predicted is False


def test_description_html_stripped_and_unescaped():
    p = _norm(SAMPLE)
    assert "<b>" not in p.description_text
    assert "SQL & Python" in p.description_text


def test_remote_inferred_from_text():
    onsite = _norm(SAMPLE)
    assert onsite.remote is False
    remote_rec = dict(SAMPLE, title="Remote Data Analyst")
    assert _norm(remote_rec).remote is True
