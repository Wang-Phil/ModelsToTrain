#!/usr/bin/env python3
"""Merge table2_per_model/macro_rows/*.json into metrics/table2_val_macro.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SIDECAR = HERE / "table2_per_model" / "macro_rows"
OUT = HERE / "metrics" / "table2_val_macro.csv"

FIELDS = [
    "excel_model", "model", "split", "n_samples", "class_counts_match",
    "acc", "auc", "acc_delta", "auc_delta",
    "sensitivity", "specificity", "npv", "ppv",
]


def main() -> None:
    rows = []
    for p in sorted(SIDECAR.glob("*_macro.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No sidecars in {SIDECAR}")
    rows.sort(key=lambda r: float(str(r["auc"]).split("(")[0]), reverse=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Merged {len(rows)} models -> {OUT}")
    df = pd.read_csv(OUT)
    print(df[["excel_model", "acc", "auc", "acc_delta", "auc_delta"]].to_string(index=False))


if __name__ == "__main__":
    main()
