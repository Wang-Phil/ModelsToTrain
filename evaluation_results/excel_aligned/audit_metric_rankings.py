#!/usr/bin/env python3
"""
Audit CasGNet rank #1 status across all macro + per-class metrics for Table 1 and Table 2.

Usage (project root):
  python evaluation_results/excel_aligned/audit_metric_rankings.py
  python evaluation_results/excel_aligned/audit_metric_rankings.py --save-before
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from metric_ranking_utils import (  # noqa: E402
    build_rank_matrix,
    casgnet_failures,
    parse_point,
    write_rank_matrix_report,
)

TABLE1_MACRO = HERE / "table1_per_model" / "metrics" / "table1_per_model_macro.csv"
TABLE1_PC = HERE / "table1_per_model" / "metrics" / "table1_per_model_per_class.csv"
TABLE2_MACRO = HERE / "metrics" / "table2_val_macro.csv"
TABLE2_PC = HERE / "metrics" / "table2_val_per_class.csv"
TABLE2_PKG_MACRO = HERE / "table2_final_package" / "TABLE2_SUMMARY.csv"
TABLE2_PKG_PC = HERE / "table2_final_package" / "TABLE2_PER_CLASS.csv"
BEFORE_DIR = HERE / "rank_snapshots"


def load_table1() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(TABLE1_MACRO), pd.read_csv(TABLE1_PC)


def load_table2() -> tuple[pd.DataFrame, pd.DataFrame]:
    if TABLE2_MACRO.is_file():
        macro = pd.read_csv(TABLE2_MACRO)
        pc = pd.read_csv(TABLE2_PC) if TABLE2_PC.is_file() else pd.DataFrame()
        return macro, pc
    summ = pd.read_csv(TABLE2_PKG_MACRO)
    macro = summ.rename(columns={"repro_acc": "acc", "repro_auc": "auc"})
    macro["excel_model"] = macro["model"]
    pc = pd.read_csv(TABLE2_PKG_PC)
    return macro, pc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-before", action="store_true", help="Write rank_snapshots/*_before.csv")
    ap.add_argument("--compare-after", action="store_true", help="Compare with rank_snapshots/*_after.csv if present")
    args = ap.parse_args()

    t1m, t1pc = load_table1()
    t2m, t2pc = load_table2()

    t1_rm = build_rank_matrix(t1m, t1pc, model_col="excel_model", class_col="experiment")
    t2_rm = build_rank_matrix(t2m, t2pc, model_col="excel_model", class_col="model")

    if args.save_before:
        BEFORE_DIR.mkdir(parents=True, exist_ok=True)
        t1_rm.to_csv(BEFORE_DIR / "table1_before.csv", index=False)
        t2_rm.to_csv(BEFORE_DIR / "table2_before.csv", index=False)
        print(f"Saved before snapshots to {BEFORE_DIR}")

    t1_fail = casgnet_failures(t1_rm)
    t2_fail = casgnet_failures(t2_rm)
    print(f"\nTable1: CasGNet NOT #1 on {len(t1_fail)} / {len(t1_rm)} metrics")
    for _, r in t1_fail.sort_values("casgnet_rank").iterrows():
        print(f"  {r['metric']}: #{int(r['casgnet_rank'])} ({r['casgnet_value']:.4f})")

    print(f"\nTable2: CasGNet NOT #1 on {len(t2_fail)} / {len(t2_rm)} metrics")
    for _, r in t2_fail.sort_values("casgnet_rank").iterrows():
        print(f"  {r['metric']}: #{int(r['casgnet_rank'])} ({r['casgnet_value']:.4f})")

    before_t1 = pd.read_csv(BEFORE_DIR / "table1_before.csv") if (BEFORE_DIR / "table1_before.csv").is_file() else t1_rm
    before_t2 = pd.read_csv(BEFORE_DIR / "table2_before.csv") if (BEFORE_DIR / "table2_before.csv").is_file() else t2_rm
    after_t1 = pd.read_csv(BEFORE_DIR / "table1_after.csv") if args.compare_after and (BEFORE_DIR / "table1_after.csv").is_file() else None
    after_t2 = pd.read_csv(BEFORE_DIR / "table2_after.csv") if args.compare_after and (BEFORE_DIR / "table2_after.csv").is_file() else None

    write_rank_matrix_report(HERE / "RANK_MATRIX_TABLE1.md", table_name="Table 1", before=before_t1, after=after_t1)
    write_rank_matrix_report(HERE / "RANK_MATRIX_TABLE2.md", table_name="Table 2", before=before_t2, after=after_t2)
    print(f"\nReports: {HERE / 'RANK_MATRIX_TABLE1.md'}, {HERE / 'RANK_MATRIX_TABLE2.md'}")

    summary = {
        "table1_failures": len(t1_fail),
        "table2_failures": len(t2_fail),
        "table1_not_rank1": t1_fail["metric"].tolist(),
        "table2_not_rank1": t2_fail["metric"].tolist(),
    }
    (HERE / "rank_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
