"""Tests for skill extraction — especially the ambiguous cases.

Each test documents a real matching hazard the extractor has to get right.
"""
from src.skills import default_extractor as ex


def found(text):
    return set(ex.extract(text))


# --- straightforward matches ---
def test_basic_skills():
    s = found("Looking for Python, SQL and Tableau experience.")
    assert {"Python", "SQL", "Tableau"} <= s


def test_alias_maps_to_canonical():
    assert "PostgreSQL" in found("must know Postgres")
    assert "AWS" in found("experience with Amazon Web Services")
    assert "Spark" in found("PySpark pipelines in production")


def test_dedup_single_entry_per_skill():
    assert ex.extract("Python Python python").count("Python") == 1


# --- word-boundary hazards ---
def test_java_does_not_fire_on_javascript():
    assert "Java" not in found("strong JavaScript skills")
    assert "JavaScript" in found("strong JavaScript skills")


def test_sql_not_matched_inside_mysql_or_nosql():
    assert found("MySQL and NoSQL stores") == {"MySQL", "NoSQL"}


def test_excel_not_matched_inside_excellent():
    assert "Excel" not in found("an excellent and excelling candidate")


def test_excel_matches_real_mention():
    assert "Excel" in found("advanced Excel and pivot tables")
    assert "Excel" in found("Microsoft Excel required")


# --- special characters ---
def test_cpp_and_csharp():
    s = found("C++ or C# a plus, ending with Python.")
    assert {"C++", "C#", "Python"} <= s


def test_trailing_period_still_matches():
    assert "Python" in found("Primary language is Python.")


# --- ambiguous single-letter language: R ---
def test_r_positive_contexts():
    assert "R" in found("experience in R and Python")
    assert "R" in found("R, Python, SQL")
    assert "R" in found("modeling in Python/R")
    assert "R" in found("RStudio and tidyverse")


def test_r_does_not_fire_on_rnd():
    assert "R" not in found("join our R&D team")
    assert "R" not in found("the R3 storage bucket")


# --- ambiguous: Go ---
def test_go_positive_and_negative():
    assert "Go" in found("services written in Golang")
    assert "Go" in found("backend in Go and Python")
    assert "Go" not in found("ready to go and make an impact")


# --- multi-word skills ---
def test_multiword_skills():
    s = found("Power BI dashboards and Machine Learning models")
    assert {"Power BI", "Machine Learning"} <= s
