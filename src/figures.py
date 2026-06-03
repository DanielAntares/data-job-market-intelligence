"""Build the Phase 3 charts.

Each ``build_*`` function returns a matplotlib Figure (no side effects), so the
notebook can render them inline. ``main()`` sets a headless backend and writes
them as PNGs under ``reports/figures/`` for the README.

    python -m src.figures
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt

from . import analysis
from .storage import load_skills, load_warehouse

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("figures")

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
_ACCENT = "#2b6cb0"
_ACCENT2 = "#2f855a"


def build_demand(jobs, skills, top: int = 15, rich_only: bool = False):
    dem = analysis.demand_table(jobs, skills, rich_only=rich_only).head(top).iloc[::-1]
    base_n = len(analysis.rich_postings(jobs)) if rich_only else len(jobs)
    note = "detailed postings" if rich_only else "postings"
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(dem["skill"], dem["pct_of_postings"], color=_ACCENT)
    for y, (pct, n) in enumerate(zip(dem["pct_of_postings"], dem["postings"])):
        ax.text(pct + 0.3, y, f"{pct:.0f}%  (n={n})", va="center", fontsize=8)
    ax.set_xlabel("% of postings mentioning the skill")
    ax.set_title(f"Most in-demand skills  (top {top} of {base_n} {note})")
    ax.margins(x=0.18)
    fig.tight_layout()
    return fig


def build_salary_by_skill(jobs, skills, top: int = 15):
    sal = analysis.salary_by_skill(jobs, skills).head(top).iloc[::-1]
    if sal.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(sal["skill"], sal["median_salary"], color=_ACCENT2)
    for y, (val, n) in enumerate(zip(sal["median_salary"], sal["n"])):
        ax.text(val + 1500, y, f"${val/1000:.0f}k (n={n})", va="center", fontsize=8)
    ax.set_xlabel("Median advertised salary (USD/yr)")
    ax.set_title("Pay by skill — advertised USD salaries only")
    ax.margins(x=0.2)
    fig.tight_layout()
    return fig


def build_cooccurrence(skills, top_n: int = 12):
    co = analysis.skill_cooccurrence(skills, top_n=top_n)
    if co.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(co.values, cmap="Blues")
    ax.set_xticks(range(len(co)), co.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(co)), co.index, fontsize=8)
    hot = co.values.max() * 0.6
    for i in range(len(co)):
        for j in range(len(co)):
            v = co.values[i, j]
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=7,
                        color="white" if v > hot else "black")
    ax.set_title("Skill co-occurrence (postings mentioning both)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="postings")
    fig.tight_layout()
    return fig


def build_roles(jobs):
    counts = analysis.role_counts(jobs)
    sal = analysis.salary_by_role(jobs).set_index("role")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(counts.index, counts.values, color=_ACCENT)
    ax1.set_title("Postings by role")
    ax1.set_ylabel("postings")
    ax1.set_ylim(0, counts.max() * 1.15)  # headroom so labels stay inside the box
    ax1.tick_params(axis="x", rotation=30)
    for x, v in enumerate(counts.values):
        ax1.text(x, v + counts.max() * 0.01, str(v), ha="center", va="bottom", fontsize=8)

    if not sal.empty:
        top = sal["median_salary"].max()
        ax2.bar(sal.index, sal["median_salary"], color=_ACCENT2)
        ax2.set_title("Median advertised salary by role (USD/yr)")
        ax2.set_ylim(0, top * 1.22)  # extra headroom for the two-line labels
        for x, (v, n) in enumerate(zip(sal["median_salary"], sal["n"])):
            ax2.text(x, v + top * 0.01, f"${v/1000:.0f}k\n(n={n})",
                     ha="center", va="bottom", fontsize=8)
        ax2.tick_params(axis="x", rotation=30)
    else:
        ax2.set_visible(False)
    fig.tight_layout()
    return fig


_FIGURES = {
    "demand_top_skills.png": lambda j, s: build_demand(j, s),
    "salary_by_skill.png": lambda j, s: build_salary_by_skill(j, s),
    "skill_cooccurrence.png": lambda j, s: build_cooccurrence(s),
    "roles.png": lambda j, s: build_roles(j),
}


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless for CLI runs

    jobs = load_warehouse()
    skills = load_skills()
    if jobs.empty or skills.empty:
        log.error("Need data — run `python -m src.collect` then `python -m src.extract_skills`.")
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Rendering charts to reports/figures/ ...")
    for name, builder in _FIGURES.items():
        fig = builder(jobs, skills)
        if fig is None:
            log.warning("  skipped %s (insufficient data)", name)
            continue
        fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        log.info("  wrote reports/figures/%s", name)
    log.info("Done.")


if __name__ == "__main__":
    main()
