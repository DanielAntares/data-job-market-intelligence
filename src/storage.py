"""Persist collected postings.

Two layers, on purpose:

1. **Raw snapshots** (``data/raw/<source>_<timestamp>.jsonl``) — the exact API
   payloads, never mutated. This is your audit trail: if you later change how a
   field is parsed, you can replay history instead of losing it.

2. **Warehouse** (``data/warehouse/jobs.parquet``) — the deduplicated,
   normalized table everything downstream reads. Re-running the pipeline merges
   new postings in, tracking ``first_seen_at`` / ``last_seen_at`` so you can
   analyze how long roles stay open once you've collected for a while.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .schema import COLUMNS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_PATH = DATA_DIR / "warehouse" / "jobs.parquet"
SKILLS_PATH = DATA_DIR / "warehouse" / "job_skills.parquet"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_raw_snapshot(source: str, records: list[dict]) -> Path:
    """Dump untouched API records as line-delimited JSON."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{source}_{stamp}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def merge_into_warehouse(rows: list[dict]) -> pd.DataFrame:
    """Upsert normalized rows into the parquet warehouse; return the full table.

    Dedup key is ``job_id``. For repeat sightings we keep the earliest
    ``first_seen_at`` and bump ``last_seen_at`` to now.
    """
    WAREHOUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = _utc_now()

    incoming = pd.DataFrame(rows, columns=COLUMNS)
    incoming["first_seen_at"] = now
    incoming["last_seen_at"] = now

    if WAREHOUSE_PATH.exists():
        existing = pd.read_parquet(WAREHOUSE_PATH)
        combined = pd.concat([existing, incoming], ignore_index=True)
    else:
        combined = incoming

    # tags is a list column -> not hashable; sort/drop_duplicates need care.
    combined = combined.sort_values("last_seen_at")
    first_seen = combined.groupby("job_id")["first_seen_at"].min()
    combined = combined.drop_duplicates(subset="job_id", keep="last").copy()
    combined["first_seen_at"] = combined["job_id"].map(first_seen)

    combined.to_parquet(WAREHOUSE_PATH, index=False)
    return combined


def load_warehouse() -> pd.DataFrame:
    if WAREHOUSE_PATH.exists():
        return pd.read_parquet(WAREHOUSE_PATH)
    return pd.DataFrame(columns=COLUMNS + ["first_seen_at", "last_seen_at"])


# --- Skills table (tidy: one row per job_id x skill) -------------------------

def save_skills(df: pd.DataFrame) -> Path:
    SKILLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SKILLS_PATH, index=False)
    return SKILLS_PATH


def load_skills() -> pd.DataFrame:
    if SKILLS_PATH.exists():
        return pd.read_parquet(SKILLS_PATH)
    return pd.DataFrame(columns=["job_id", "skill", "category"])
