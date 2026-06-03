"""Tests for short-field cleaning (HTML entity decoding, whitespace)."""
from src.sources.base import clean_field


def test_decodes_ampersand_entity():
    assert clean_field("Data Science &amp; Analytics") == "Data Science & Analytics"


def test_decodes_numeric_entity():
    assert clean_field("Caf&#233;") == "Café"


def test_collapses_whitespace():
    assert clean_field("  Senior   Data    Analyst ") == "Senior Data Analyst"


def test_handles_none_and_empty():
    assert clean_field(None) == ""
    assert clean_field("") == ""
