"""Unit tests for the messy-salary parser.

These double as documentation of exactly which real-world formats we handle.
Run with:  pytest
"""
from src.salary import parse_salary


def test_k_range_yearly():
    out = parse_salary("$80k - $100k")
    assert out["salary_min"] == 80_000
    assert out["salary_max"] == 100_000
    assert out["salary_currency"] == "USD"
    assert out["salary_period"] == "yearly"


def test_hourly_explicit():
    out = parse_salary("$18 - $22/hr")
    assert out["salary_min"] == 18
    assert out["salary_max"] == 22
    assert out["salary_period"] == "hourly"


def test_hourly_with_spaces():
    out = parse_salary("$90 - $150 /hour")
    assert out["salary_min"] == 90
    assert out["salary_max"] == 150
    assert out["salary_period"] == "hourly"


def test_full_numbers_with_commas():
    out = parse_salary("$80,000 - $100,000 per year")
    assert out["salary_min"] == 80_000
    assert out["salary_max"] == 100_000
    assert out["salary_period"] == "yearly"


def test_single_value():
    out = parse_salary("$120,000")
    assert out["salary_min"] == 120_000
    assert out["salary_max"] is None
    assert out["salary_period"] == "yearly"


def test_euro_currency():
    out = parse_salary("€50k")
    assert out["salary_currency"] == "EUR"
    assert out["salary_min"] == 50_000


def test_shorthand_range_only_upper_has_k():
    # "$80-100k" should lift the lower bound to the same magnitude.
    out = parse_salary("$80-100k")
    assert out["salary_min"] == 80_000
    assert out["salary_max"] == 100_000


def test_magnitude_infers_hourly():
    out = parse_salary("$25 - $40")  # no period word, small numbers
    assert out["salary_period"] == "hourly"


def test_empty_and_none():
    for val in (None, "", "Competitive", "DOE"):
        out = parse_salary(val)
        assert out["salary_min"] is None
        assert out["salary_max"] is None
