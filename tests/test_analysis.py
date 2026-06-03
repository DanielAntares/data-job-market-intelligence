"""Tests for the analysis tables (role classification, demand, salary, co-occ.)."""
import pandas as pd

from src import analysis


def test_classify_role_specific_before_generic():
    assert analysis.classify_role("Senior Data Engineer") == "Data Engineer"
    assert analysis.classify_role("Machine Learning Engineer") == "ML Engineer"
    assert analysis.classify_role("Staff Data Scientist") == "Data Scientist"
    assert analysis.classify_role("Marketing Data Analyst") == "Data Analyst"
    assert analysis.classify_role("Product Manager") == "Other"


def test_classify_role_engineer_beats_analyst_keyword():
    # contains "analytics" but is clearly an engineering role -> Engineer wins
    assert analysis.classify_role("Analytics Engineer") == "Data Engineer"


def _jobs():
    return pd.DataFrame({
        "job_id": ["a", "b", "c"],
        "title": ["Data Analyst", "Data Scientist", "Data Engineer"],
        "salary_min": [90000, 150000, 130000],
        "salary_max": [110000, 170000, 130000],
        "salary_period": ["yearly", "yearly", "yearly"],
        "salary_currency": ["USD", "USD", "USD"],
        "salary_is_predicted": [False, False, True],  # c is predicted -> excluded
    })


def _skills():
    return pd.DataFrame({
        "job_id": ["a", "a", "b", "b", "c"],
        "skill": ["SQL", "Excel", "SQL", "Python", "Python"],
        "category": ["language", "viz_bi", "language", "language", "language"],
    })


def test_demand_counts_and_pct():
    dem = analysis.demand_table(_jobs(), _skills())
    sql = dem[dem["skill"] == "SQL"].iloc[0]
    assert sql["postings"] == 2
    assert sql["pct_of_postings"] == round(100 * 2 / 3, 1)


def test_salary_table_excludes_predicted():
    sal = analysis.build_salary_table(_jobs())
    assert set(sal["job_id"]) == {"a", "b"}  # c excluded (predicted)
    # midpoint of a's range
    assert sal[sal.job_id == "a"]["salary"].iloc[0] == 100000


def test_cooccurrence_diagonal_is_skill_count():
    co = analysis.skill_cooccurrence(_skills(), top_n=5)
    assert co.loc["SQL", "SQL"] == 2
    assert co.loc["SQL", "Python"] == 1  # co-occur only in posting b


def test_demand_rich_only_excludes_truncated_descriptions():
    jobs = pd.DataFrame({
        "job_id": ["rich", "thin"],
        "title": ["Data Analyst", "Data Analyst"],
        "description_text": ["x" * 1000, "short teaser"],  # thin = truncated
    })
    skills = pd.DataFrame({
        "job_id": ["rich", "thin"],
        "skill": ["SQL", "SQL"],
        "category": ["language", "language"],
    })
    # Opt-in rich filter: only the rich posting counts.
    dem = analysis.demand_table(jobs, skills, rich_only=True, min_chars=600)
    sql = dem[dem["skill"] == "SQL"].iloc[0]
    assert sql["postings"] == 1
    assert sql["pct_of_postings"] == 100.0

    # Default (rich_only=False): both postings count -> 2/2.
    dem_all = analysis.demand_table(jobs, skills)
    assert dem_all[dem_all["skill"] == "SQL"].iloc[0]["postings"] == 2


def test_quality_by_source_reports_coverage():
    jobs = pd.DataFrame({
        "job_id": ["a", "b", "c"],
        "source": ["jsearch", "jsearch", "adzuna"],
        "title": ["Data Analyst"] * 3,
        "description_text": ["x" * 2000, "y" * 1500, "z" * 400],
    })
    skills = pd.DataFrame({
        "job_id": ["a", "b"], "skill": ["SQL", "Python"],
        "category": ["language", "language"],
    })
    q = analysis.quality_by_source(jobs, skills).set_index("source")
    assert q.loc["jsearch", "pct_with_skill"] == 100   # both jsearch posts have a skill
    assert q.loc["adzuna", "pct_with_skill"] == 0       # adzuna post has none
    assert q.loc["adzuna", "avg_desc_chars"] == 400
