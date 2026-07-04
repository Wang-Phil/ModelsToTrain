#!/usr/bin/env python3
"""
Export Table 1 and Table 2 final packages to Excel workbooks.

Usage (project root):
  python evaluation_results/excel_aligned/export_all_tables_excel.py

Writes:
  table1_final_package/TABLE1_RESULTS.xlsx
  table2_final_package/TABLE2_RESULTS.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from export_table1_excel import export_excel as export_table1
from export_table2_excel import export_excel as export_table2

HERE = Path(__file__).resolve().parent
TABLE1_XLSX = HERE / "table1_final_package" / "TABLE1_RESULTS.xlsx"
TABLE2_XLSX = HERE / "table2_final_package" / "TABLE2_RESULTS.xlsx"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export all table packages to Excel.")
    parser.add_argument(
        "--table1-only",
        action="store_true",
        help="Export only Table 1",
    )
    parser.add_argument(
        "--table2-only",
        action="store_true",
        help="Export only Table 2",
    )
    args = parser.parse_args()

    export_t1 = not args.table2_only
    export_t2 = not args.table1_only

    if export_t1:
        counts = export_table1(TABLE1_XLSX)
        print(f"Wrote {TABLE1_XLSX}")
        for sheet, n in counts.items():
            print(f"  {sheet}: {n} rows")

    if export_t2:
        counts = export_table2(TABLE2_XLSX)
        print(f"Wrote {TABLE2_XLSX}")
        for sheet, n in counts.items():
            print(f"  {sheet}: {n} rows")


if __name__ == "__main__":
    main()
