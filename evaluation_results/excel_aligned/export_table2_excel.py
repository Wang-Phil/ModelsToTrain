#!/usr/bin/env python3
"""
Export Table 2 (表二) summary and per-class metrics to Excel.

Usage (project root):
  python evaluation_results/excel_aligned/export_table2_excel.py

Reads:
  table2_final_package/TABLE2_SUMMARY.csv
  table2_final_package/TABLE2_PER_CLASS.csv

Writes:
  table2_final_package/TABLE2_RESULTS.xlsx
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE / "table2_final_package"
SUMMARY_CSV = PKG / "TABLE2_SUMMARY.csv"
PER_CLASS_CSV = PKG / "TABLE2_PER_CLASS.csv"
OUTPUT_XLSX = PKG / "TABLE2_RESULTS.xlsx"

import re


def parse_point(s: str | float | None) -> float | None:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    m = re.match(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(SUMMARY_CSV)
    df["_repro_auc_point"] = df["repro_auc"].map(parse_point)
    df = df.sort_values("_repro_auc_point", ascending=False, kind="mergesort").drop(
        columns=["_repro_auc_point"]
    )
    df = df.reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    cols = [
        "rank",
        "model",
        "checkpoint",
        "split",
        "n",
        "repro_acc",
        "excel_acc",
        "acc_delta",
        "repro_auc",
        "excel_auc",
        "auc_delta",
        "match_status",
        "sensitivity",
        "specificity",
        "npv",
        "ppv",
    ]
    return df[[c for c in cols if c in df.columns]]


def load_per_class(model_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(PER_CLASS_CSV)
    # Source CSV stores class names in the "model" column; "class" is empty.
    if (
        "class" in df.columns
        and df["class"].isna().all()
        and "model" in df.columns
        and "excel_model" in df.columns
    ):
        df = df.drop(columns=["class"]).rename(columns={"model": "class", "excel_model": "model"})
    elif "excel_model" in df.columns:
        df = df.rename(columns={"excel_model": "model"})
    order_map = {m: i for i, m in enumerate(model_order)}
    df["_model_ord"] = df["model"].map(order_map).fillna(999).astype(int)
    df = df.sort_values(["_model_ord", "class"], kind="mergesort").drop(columns=["_model_ord"])
    cols = ["model", "split", "class", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


def build_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for metric_col, label in (("repro_auc", "AUC"), ("repro_acc", "ACC")):
        pts = [
            (r["model"], parse_point(r[metric_col]))
            for _, r in summary.iterrows()
            if parse_point(r[metric_col]) is not None
        ]
        pts.sort(key=lambda x: -x[1])
        for rank, (model, point) in enumerate(pts, start=1):
            raw = summary.loc[summary["model"] == model, metric_col].iloc[0]
            rows.append(
                {
                    "metric": label,
                    "rank": rank,
                    "model": model,
                    "value": raw,
                    "point": point,
                }
            )
    return pd.DataFrame(rows)


def export_excel(output: Path = OUTPUT_XLSX) -> dict[str, int]:
    if not SUMMARY_CSV.is_file():
        raise FileNotFoundError(f"Missing summary CSV: {SUMMARY_CSV}")
    if not PER_CLASS_CSV.is_file():
        raise FileNotFoundError(f"Missing per-class CSV: {PER_CLASS_CSV}")

    summary = load_summary()
    per_class = load_per_class(summary["model"].tolist())
    ranking = build_ranking(summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="总体指标", index=False)
        per_class.to_excel(writer, sheet_name="每类指标", index=False)
        ranking.to_excel(writer, sheet_name="排名", index=False)

    return {
        "总体指标": len(summary),
        "每类指标": len(per_class),
        "排名": len(ranking),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Table 2 metrics to Excel.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_XLSX,
        help=f"Output xlsx path (default: {OUTPUT_XLSX})",
    )
    args = parser.parse_args()

    counts = export_excel(args.output)
    print(f"Wrote {args.output}")
    for sheet, n in counts.items():
        print(f"  {sheet}: {n} rows")


if __name__ == "__main__":
    main()
