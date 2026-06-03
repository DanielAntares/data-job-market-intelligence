"""Pipeline entry point: collect -> normalize -> snapshot -> warehouse.

Run it:
    python -m src.collect                  # use config.yaml
    python -m src.collect --limit 50       # smaller pull
    python -m src.collect --sources remotive
    python -m src.collect --search "data analyst" "nlp engineer"

Designed to be safe to run repeatedly (and on a schedule): the warehouse
deduplicates, so each run just tops it up with whatever's new.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .sources import REGISTRY
from .storage import merge_into_warehouse, save_raw_snapshot

# Load API keys (e.g. Adzuna) from a git-ignored .env before sources read them.
load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("collect")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect data-job postings into the warehouse.")
    p.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to config.yaml")
    p.add_argument("--sources", nargs="*", help="Subset of sources to run (default: all enabled)")
    p.add_argument("--search", nargs="*", help="Override search terms from config")
    p.add_argument("--limit", type=int, help="Override postings-per-source limit")
    return p.parse_args()


def enabled_sources(config: dict, only: list[str] | None) -> list[str]:
    configured = {
        name for name, opts in (config.get("sources") or {}).items()
        if (opts or {}).get("enabled") and name in REGISTRY
    }
    if only:
        return [s for s in only if s in configured or s in REGISTRY]
    return sorted(configured)


def summarize(df) -> None:
    total = len(df)
    with_salary = int(df["salary_min"].notna().sum())
    log.info("-" * 56)
    log.info("Warehouse now holds %d unique postings", total)
    if total:
        pct = 100 * with_salary / total
        log.info("  with salary data: %d (%.0f%%)", with_salary, pct)
        by_source = df["source"].value_counts().to_dict()
        log.info("  by source: %s", ", ".join(f"{k}={v}" for k, v in by_source.items()))
        top = df["category"].dropna().value_counts().head(5)
        if not top.empty:
            log.info("  top categories: %s", ", ".join(f"{k} ({v})" for k, v in top.items()))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    search_terms = args.search or config.get("search_terms", [])
    limit = args.limit or config.get("limit_per_source", 100)
    sources = enabled_sources(config, args.sources)

    if not sources:
        log.error("No sources enabled. Check config.yaml.")
        return

    log.info("Collecting %r from %s (limit %d/source)", search_terms, sources, limit)

    all_rows: list[dict] = []
    for name in sources:
        source = REGISTRY[name](config.get("sources", {}).get(name, {}))
        try:
            raw, postings = source.collect(search_terms, limit)
        except Exception as exc:
            log.warning("  %s failed: %s", name, exc)
            continue
        snapshot = save_raw_snapshot(name, raw)
        rows = [p.to_row() for p in postings]
        all_rows.extend(rows)
        log.info("  %-9s %3d raw -> %3d normalized  (snapshot: %s)",
                 name, len(raw), len(rows), snapshot.name)

    if not all_rows:
        log.warning("Collected nothing this run.")
        return

    df = merge_into_warehouse(all_rows)
    summarize(df)


if __name__ == "__main__":
    main()
