#!/usr/bin/env python3
"""
Quick validation utility for Refuel config files.

Checks that the price, restock, and mix datasets exist and contain the expected schema.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Anchor the script to repo root so relative paths work when invoked anywhere.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def validate_product_prices() -> list[str]:
    issues: list[str] = []
    price_file = DATA_DIR / "product_prices.csv"
    if not price_file.exists():
        issues.append("Missing product_prices.csv. Open the Price Manager page to initialize it.")
        return issues
    # Only basic schema/NA checks are needed for this quick validation.
    df = pd.read_csv(price_file)
    required_cols = {"product", "unit_price"}
    if not required_cols.issubset(df.columns):
        issues.append(f"`product_prices.csv` missing columns: {required_cols - set(df.columns)}")
    elif df["unit_price"].isna().any():
        issues.append("`product_prices.csv` has blank unit_price values.")
    return issues


def validate_restock_policy() -> list[str]:
    issues: list[str] = []
    policy_file = DATA_DIR / "restock_policy.json"
    if not policy_file.exists():
        issues.append("Missing restock_policy.json. Configure auto restock once in the POS Console.")
        return issues
    try:
        policy = json.loads(policy_file.read_text())
    except Exception as exc:
        issues.append(f"Invalid restock_policy.json: {exc}")
        return issues
    required_keys = {"auto_enabled", "threshold_units", "lot_size", "cooldown_hours"}
    if not required_keys.issubset(policy.keys()):
        issues.append(f"`restock_policy.json` missing keys: {required_keys - set(policy.keys())}")
    return issues


def validate_mix_recency() -> list[str]:
    issues: list[str] = []
    mix_file = DATA_DIR / "product_mix_daily.csv"
    if not mix_file.exists():
        issues.append("Missing product_mix_daily.csv.")
        return issues
    # Parse the date column so freshness comparisons stay robust.
    df = pd.read_csv(mix_file, parse_dates=["date"])
    if df.empty:
        issues.append("`product_mix_daily.csv` has no rows.")
        return issues
    latest_date = df["date"].max()
    days_old = (datetime.now(timezone.utc).date() - latest_date.date()).days
    if days_old > 1:
        issues.append(f"Product mix data is {days_old} day(s) old (latest {latest_date.date()}).")
    return issues


def main() -> None:
    # Group each validation with a label so the output stays readable.
    checks = [
        ("Product prices", validate_product_prices),
        ("Restock policy", validate_restock_policy),
        ("Product mix freshness", validate_mix_recency),
    ]
    any_issue = False
    for label, fn in checks:
        issues = fn()
        if issues:
            any_issue = True
            print(f"[WARN] {label}:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"[OK] {label}")
    if not any_issue:
        print("All config files look good!")


if __name__ == "__main__":
    main()
