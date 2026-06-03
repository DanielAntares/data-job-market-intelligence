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
