"""Tests for salary-model feature engineering (the deterministic parts)."""
import pandas as pd

from src import model


def test_seniority_levels():
    assert model.seniority_level("Principal Data Scientist") == 4
    assert model.seniority_level("Lead Data Engineer") == 4
    assert model.seniority_level("Senior Data Analyst") == 3
    assert model.seniority_level("Data Analyst") == 2
    assert model.seniority_level("Junior Data Analyst") == 1
    assert model.seniority_level(None) == 2


def _jobs():
    return pd.DataFrame({
        "job_id": ["a", "b"],
        "title": ["Senior Data Scientist", "Data Analyst"],
        "remote": [True, False],
        "salary_min": [150000, 90000],
        "salary_max": [170000, 110000],
        "salary_period": ["yearly", "yearly"],
        "salary_currency": ["USD", "USD"],
        "salary_is_predicted": [False, False],
    })


def _skills():
    return pd.DataFrame({
        "job_id": ["a", "a", "a", "b", "b"],
        "skill": ["Python", "PyTorch", "Deep Learning", "SQL", "Excel"],
        "category": ["language", "ml_library", "method", "language", "viz_bi"],
    })


def test_build_features_shape_and_flags():
    feats = model.build_features(_jobs(), _skills())
    assert len(feats) == 2
    a = feats[feats["role"] == "Data Scientist"].iloc[0]
    assert a["seniority"] == 3
    assert a["remote"] == 1
    assert a["n_skills"] == 3
    assert a["has_python"] == 1
    assert a["has_deep_learning"] == 1
    assert a["has_ml_library"] == 1  # PyTorch
    assert a["has_sql"] == 0

    b = feats[feats["role"] == "Data Analyst"].iloc[0]
    assert b["has_sql"] == 1
    assert b["has_ml_library"] == 0
    assert b["salary"] == 100000  # midpoint of advertised range


def test_build_features_excludes_predicted_salary():
    jobs = _jobs()
    jobs.loc[jobs.job_id == "b", "salary_is_predicted"] = True
    feats = model.build_features(jobs, _skills())
    assert len(feats) == 1  # only the advertised one survives


def test_location_bucketing():
    assert model.location_bucket("USA") == "US"
    assert model.location_bucket("Sydney, Sydney Region") == "ANZ"
    assert model.location_bucket("EMEA") == "Europe"
    assert model.location_bucket("Jakarta, Indonesia") == "Asia"
    assert model.location_bucket("Worldwide") == "Worldwide"
    assert model.location_bucket(None) == "Other"


def test_years_of_experience_reads_the_common_phrasings():
    assert model.years_of_experience("We need 5+ years of experience") == 5
    assert model.years_of_experience("minimum of 3 years in analytics") == 3
    assert model.years_of_experience("4-6 years building pipelines") == 4
    assert model.years_of_experience("7 years of hands-on experience") == 7
    # nothing stated -> NaN, so the imputer (not a made-up zero) handles it
    assert pd.isna(model.years_of_experience("A great place to work"))
    # implausible figures are ignored rather than trusted
    assert pd.isna(model.years_of_experience("founded 99 years ago"))


def test_scrub_text_removes_any_quoted_salary():
    """TF-IDF must not be able to read the target back out of the description."""
    scrubbed = model.scrub_text("Pay is $150,000 - $180,000 per year")
    assert "150" not in scrubbed and "$" not in scrubbed
    assert "year" in scrubbed


def test_normalize_company_folds_legal_suffixes():
    assert model.normalize_company("ManTech International") == "mantech"
    assert model.normalize_company("ManTech") == "mantech"
    assert model.normalize_company("Acme Technologies, Inc.") == "acme"
    assert model.normalize_company(None) == "(unknown)"


def test_build_features_carries_company_for_grouped_cv():
    jobs = _jobs()
    jobs["company"] = ["Acme Inc.", "Acme"]
    feats = model.build_features(jobs, _skills())
    # both postings must land in one CV group, or the split leaks
    assert feats["company"].nunique() == 1


def test_predict_salary_responds_to_location():
    """Sanity check that the live model actually uses the new inputs."""
    jobs = pd.DataFrame({
        "job_id": [str(i) for i in range(12)],
        "title": ["Senior Data Scientist", "Data Analyst"] * 6,
        "remote": [True, False] * 6,
        "location": ["USA"] * 6 + ["Indonesia"] * 6,
        "description_text": ["5+ years of experience. " + "x" * 700] * 12,
        "company": [f"co{i}" for i in range(12)],
        "salary_min": [180000, 140000] * 3 + [40000, 30000] * 3,
        "salary_max": [200000, 160000] * 3 + [50000, 40000] * 3,
        "salary_period": ["yearly"] * 12,
        "salary_currency": ["USD"] * 12,
        "salary_is_predicted": [False] * 12,
    })
    skills = pd.DataFrame({
        "job_id": [str(i) for i in range(12)],
        "skill": ["Python"] * 12,
        "category": ["language"] * 12,
    })
    pipe, n = model.train_salary_model(jobs, skills)
    assert n == 12
    us = model.predict_salary(pipe, "Data Scientist", 3, True, {"Python"},
                              location="US")
    asia = model.predict_salary(pipe, "Data Scientist", 3, True, {"Python"},
                                location="Asia")
    assert us > asia
