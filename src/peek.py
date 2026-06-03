"""Quick read-only look at the warehouse — sanity-check after collecting.

    python -m src.peek            # summary + a few sample rows
    python -m src.peek --sample 10
    python -m src.peek --salary   # only rows with a real advertised salary
"""
from __future__ import annotations

import argparse

import pandas as pd

from .storage import load_warehouse

pd.set_option("display.max_colwidth", 40)
pd.set_option("display.width", 140)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect the jobs warehouse.")
    ap.add_argument("--sample", type=int, default=5, help="How many sample rows to show")
    ap.add_argument("--salary", action="store_true", help="Only rows with a real advertised salary")
    args = ap.parse_args()

    df = load_warehouse()
    if df.empty:
        print("Warehouse is empty — run `python -m src.collect` first.")
        return

    has_sal = df["salary_min"].notna()
    predicted = df["salary_is_predicted"].fillna(False).astype(bool)
    advertised = has_sal & ~predicted

    print(f"TOTAL POSTINGS : {len(df)}")
    print(f"  by source    : " + ", ".join(f"{k}={v}" for k, v in df['source'].value_counts().items()))
    print(f"  remote/onsite: {int(df['remote'].sum())} remote, {int((~df['remote']).sum())} on-site/hybrid")
    print(f"  with salary  : {int(has_sal.sum())}  "
          f"(advertised={int(advertised.sum())}, predicted={int((has_sal & predicted).sum())})")
    print(f"  unique firms : {df['company'].nunique()}")

    top = df["category"].dropna().value_counts().head(5)
    if not top.empty:
        print("  top categories: " + ", ".join(f"{k} ({v})" for k, v in top.items()))

    view = df[advertised] if args.salary else df
    cols = ["source", "title", "company", "location", "remote",
            "salary_min", "salary_max", "salary_currency", "salary_period"]
    print(f"\nSAMPLE ROWS ({'advertised-salary only' if args.salary else 'all'}):")
    print(view[cols].head(args.sample).to_string(index=False))


if __name__ == "__main__":
    main()
