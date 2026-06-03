"""Turn the warehouse + skills table into the headline insights.

    python -m src.report

Two questions it answers:
  1. **Demand** — which skills appear in the most postings?
  2. **Pay** — among postings with a *genuinely advertised* USD annual salary,
     which skills are associated with higher median pay?

Salary analysis is deliberately strict: advertised (not Adzuna-predicted),
annual, USD only — so we compare like with like. Small-sample skills are hidden
behind ``MIN_N`` because a median over 2 postings is noise, not signal.
"""
from __future__ import annotations

from .analysis import (
    MIN_N,
    build_salary_table,
    demand_table,
    quality_by_source,
    salary_by_role,
    salary_by_skill,
)
from .storage import load_skills, load_warehouse


def print_report(top: int = 15) -> None:
    jobs = load_warehouse()
    skills = load_skills()
    if jobs.empty or skills.empty:
        print("Need data first: run `python -m src.collect` then `python -m src.extract_skills`.")
        return

    n_jobs = len(jobs)
    covered = skills["job_id"].nunique()
    print("=" * 60)
    print(f"DATA JOB MARKET — INSIGHTS  ({n_jobs} postings, "
          f"{covered} with >=1 detected skill)")
    print("=" * 60)

    print(f"\nMOST IN-DEMAND SKILLS (top {top}; % of all {n_jobs} postings)")
    dem = demand_table(jobs, skills).head(top)
    for _, r in dem.iterrows():
        bar = "#" * int(r["pct_of_postings"] / 2)
        print(f"  {r['skill']:<18} {r['postings']:>4}  {r['pct_of_postings']:>5.1f}%  {bar}")

    print(f"\nMEDIAN ADVERTISED SALARY BY SKILL  (USD/yr, n>={MIN_N})")
    sal = salary_by_skill(jobs, skills)
    if sal.empty:
        print(f"  (not enough advertised-salary postings yet — keep collecting)")
    else:
        for _, r in sal.iterrows():
            print(f"  {r['skill']:<18} ${r['median_salary']:>8,}  (n={r['n']})")

    roles = salary_by_role(jobs)
    if not roles.empty:
        print(f"\nMEDIAN ADVERTISED SALARY BY ROLE  (USD/yr, n>={MIN_N})")
        for _, r in roles.iterrows():
            print(f"  {r['role']:<18} ${r['median_salary']:>8,}  (n={r['n']})")

    print(f"\nDATA QUALITY BY SOURCE")
    print(f"  {'source':<10}{'posts':>6}{'avg_desc':>10}{'%_skill':>9}{'skills/post':>13}")
    for _, r in quality_by_source(jobs, skills).iterrows():
        print(f"  {r['source']:<10}{int(r['posts']):>6}{int(r['avg_desc_chars']):>10}"
              f"{int(r['pct_with_skill']):>8}%{r['avg_skills']:>13}")
    print("  (Adzuna returns truncated ~500-char teasers, so its low skill counts are")
    print("   missing data, not low demand -- weigh the rich-description sources more.)")

    sal_n = len(build_salary_table(jobs))
    print(f"\nNote: salary figures use {sal_n} postings with a genuinely advertised "
          f"USD annual salary\n(Adzuna-predicted salaries excluded). Numbers firm up as data accumulates.")


if __name__ == "__main__":
    print_report()
