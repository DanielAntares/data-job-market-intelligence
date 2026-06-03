"""Analytical tables built from the warehouse + skills table.

Pure pandas, no plotting and no printing — so the functions are easy to unit
test and get reused by both the text report ([report.py](report.py)) and the
charts ([figures.py](figures.py)).

Salary rule (used everywhere): only *advertised* (not Adzuna-predicted),
annual, USD salaries, so figures compare like with like.
"""
from __future__ import annotations

import pandas as pd

MIN_N = 5  # minimum postings before a per-group salary median is trustworthy

# Minimum description length for a posting to be trustworthy for *skill demand*.
# Some sources (notably Adzuna) return truncated ~500-char teasers, so a skill's
# absence there is missing data, not a true "not required". We compute demand
# only over postings whose description is substantial enough to list skills.
RICH_DESC_CHARS = 600

# Role buckets, checked in order (most specific first). A posting's title is
# matched against these keyword sets to compare Analyst vs Scientist vs Engineer.
_ROLE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Data Engineer", ("data engineer", "analytics engineer", "data platform",
                        "etl developer", "data infrastructure")),
    ("ML Engineer", ("machine learning engineer", "ml engineer", "ai engineer",
                     "mlops", "ml ops", "deep learning")),
    ("Data Scientist", ("data scientist", "data science", "research scientist",
                        "applied scientist", "decision scientist")),
    ("Data Analyst", ("data analyst", "business intelligence", "bi analyst",
                      "analytics analyst", "reporting analyst", "analyst",
                      "analytics")),
]


def classify_role(title: str | None) -> str:
    """Bucket a posting title into a coarse role family."""
    t = (title or "").lower()
    for role, keywords in _ROLE_RULES:
        if any(k in t for k in keywords):
            return role
    return "Other"


def _representative_salary(row) -> float | None:
    lo, hi = row["salary_min"], row["salary_max"]
    if pd.notna(lo) and pd.notna(hi) and hi > 0:
        return (lo + hi) / 2
    return lo if pd.notna(lo) else None


def build_salary_table(jobs: pd.DataFrame) -> pd.DataFrame:
    """One row per posting with a clean, comparable advertised salary."""
    predicted = jobs["salary_is_predicted"].fillna(False).astype(bool)
    mask = (
        jobs["salary_min"].notna()
        & ~predicted
        & (jobs["salary_period"] == "yearly")
        & (jobs["salary_currency"] == "USD")
    )
    sal = jobs.loc[mask, ["job_id", "salary_min", "salary_max"]].copy()
    sal["salary"] = sal.apply(_representative_salary, axis=1)
    return sal.dropna(subset=["salary"])[["job_id", "salary"]]


def rich_postings(jobs: pd.DataFrame, min_chars: int = RICH_DESC_CHARS) -> pd.DataFrame:
    """Postings whose description is long enough to trust skill *absence*."""
    if "description_text" not in jobs.columns:
        return jobs
    return jobs[jobs["description_text"].fillna("").str.len() >= min_chars]


def demand_table(jobs: pd.DataFrame, skills: pd.DataFrame, rich_only: bool = False,
                 min_chars: int = RICH_DESC_CHARS) -> pd.DataFrame:
    """Skill -> number and % of postings mentioning it, most-demanded first.

    Counts every posting by default. Pass ``rich_only=True`` to restrict to
    postings with a substantial description (so truncated teasers like Adzuna,
    where a missing skill is *missing data*, don't skew the percentages) — this
    is offered as an opt-in lens, not forced, since it also drops legitimately
    short-but-complete postings.
    """
    base = rich_postings(jobs, min_chars) if rich_only else jobs
    total = max(1, len(base))
    s = skills[skills["job_id"].isin(set(base["job_id"]))]
    counts = s.groupby("skill")["job_id"].nunique().rename("postings")
    out = counts.reset_index().sort_values("postings", ascending=False)
    out["pct_of_postings"] = (100 * out["postings"] / total).round(1)
    return out.reset_index(drop=True)


def quality_by_source(jobs: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    """Per-source data-quality audit: description length and skill coverage."""
    j = jobs.copy()
    j["desc_len"] = j.get("description_text", "").fillna("").str.len()
    n = skills.groupby("job_id").size().rename("n_skills")
    j = j.merge(n, on="job_id", how="left")
    j["n_skills"] = j["n_skills"].fillna(0)
    out = j.groupby("source").apply(lambda g: pd.Series({
        "posts": len(g),
        "avg_desc_chars": int(g["desc_len"].mean()),
        "pct_with_skill": round(100 * (g["n_skills"] > 0).mean()),
        "avg_skills": round(g["n_skills"].mean(), 1),
    }), include_groups=False).reset_index()
    return out.sort_values("avg_skills", ascending=False).reset_index(drop=True)


def salary_by_skill(jobs: pd.DataFrame, skills: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    """Skill -> median advertised salary, restricted to skills with >= min_n."""
    sal = build_salary_table(jobs)
    merged = skills.merge(sal, on="job_id")
    grp = merged.groupby("skill")["salary"].agg(n="count", median_salary="median")
    grp = grp[grp["n"] >= min_n].sort_values("median_salary", ascending=False)
    grp["median_salary"] = grp["median_salary"].round(0).astype(int)
    return grp.reset_index()


def with_roles(jobs: pd.DataFrame) -> pd.DataFrame:
    out = jobs.copy()
    out["role"] = out["title"].map(classify_role)
    return out


def role_counts(jobs: pd.DataFrame) -> pd.Series:
    return with_roles(jobs)["role"].value_counts()


def salary_by_role(jobs: pd.DataFrame, min_n: int = MIN_N) -> pd.DataFrame:
    j = with_roles(jobs)
    sal = build_salary_table(j)
    j = j.merge(sal, on="job_id")
    grp = j.groupby("role")["salary"].agg(n="count", median_salary="median")
    grp = grp[grp["n"] >= min_n].sort_values("median_salary", ascending=False)
    grp["median_salary"] = grp["median_salary"].round(0).astype(int)
    return grp.reset_index()


def skill_cooccurrence(skills: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Square matrix: how often each pair of the top-N skills co-occur.

    The diagonal is each skill's own posting count.
    """
    top = (
        skills.groupby("skill")["job_id"].nunique()
        .sort_values(ascending=False).head(top_n).index.tolist()
    )
    sub = skills[skills["skill"].isin(top)]
    if sub.empty:
        return pd.DataFrame(index=top, columns=top, dtype=int).fillna(0)
    present = (pd.crosstab(sub["job_id"], sub["skill"]) > 0).astype(int)
    co = present.T.dot(present)
    return co.reindex(index=top, columns=top).fillna(0).astype(int)


def top_skills_by_role(jobs: pd.DataFrame, skills: pd.DataFrame, per_role: int = 5) -> pd.DataFrame:
    """For each role, its most-demanded skills (long format)."""
    j = with_roles(jobs)[["job_id", "role"]]
    merged = skills.merge(j, on="job_id")
    counts = (
        merged.groupby(["role", "skill"])["job_id"].nunique()
        .rename("postings").reset_index()
    )
    counts = counts.sort_values(["role", "postings"], ascending=[True, False])
    return counts.groupby("role").head(per_role).reset_index(drop=True)
