"""Phase 2: extract skills from posting descriptions into a tidy table.

    python -m src.extract_skills

Reads the jobs warehouse, runs the skill extractor over each description, and
writes ``data/warehouse/job_skills.parquet`` (one row per job_id x skill).
Then prints the demand + salary report.
"""
from __future__ import annotations

import logging

import pandas as pd

from .report import print_report
from .skills import default_extractor
from .storage import load_warehouse, save_skills

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extract_skills")


def extract_all(jobs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for job_id, text in zip(jobs["job_id"], jobs["description_text"]):
        for skill, category in default_extractor.extract_with_categories(text):
            rows.append({"job_id": job_id, "skill": skill, "category": category})
    return pd.DataFrame(rows, columns=["job_id", "skill", "category"])


def main() -> None:
    jobs = load_warehouse()
    if jobs.empty:
        log.error("Warehouse is empty — run `python -m src.collect` first.")
        return

    log.info("Extracting skills from %d postings...", len(jobs))
    skills = extract_all(jobs)
    save_skills(skills)

    covered = skills["job_id"].nunique()
    per_post = len(skills) / max(1, covered)
    log.info("Found %d skill mentions across %d/%d postings (avg %.1f skills/posting).",
             len(skills), covered, len(jobs), per_post)
    log.info("Wrote data/warehouse/job_skills.parquet")
    print()
    print_report()


if __name__ == "__main__":
    main()
