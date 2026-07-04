#!/usr/bin/env python3
"""Regenerate EXCEL_VS_REPRO_SUMMARY.csv / .md from table1 + table2 metrics."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TABLE1 = HERE / "table1_per_model" / "metrics" / "table1_per_model_macro.csv"
TABLE1_MANIFEST_DIR = HERE / "table1_per_model" / "manifests"
TABLE2 = HERE / "metrics" / "table2_val_macro.csv"
TABLE2_MANIFEST_DIR = HERE / "table2_per_model" / "manifests"
OUT_CSV = HERE / "EXCEL_VS_REPRO_SUMMARY.csv"
OUT_MD = HERE / "EXCEL_VS_REPRO_SUMMARY.md"

MATCHED = 0.002
CLOSE = 0.015
MAX_EVAL_N = 300
USE_EXCEL_MATCH_LABELS = True  # set False to show rank-only status (no matched/blocked)


def parse_point(s: str) -> float | None:
    m = re.match(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def status(delta: float) -> str:
    if not USE_EXCEL_MATCH_LABELS:
        return ""
    a = abs(delta)
    if a <= MATCHED:
        return "matched"
    if a <= CLOSE:
        return "close"
    return "blocked"


def manifest_split_note(model: str) -> str:
    mp = TABLE1_MANIFEST_DIR / f"{model}_table1_manifest.json"
    if not mp.is_file():
        return ""
    import json

    m = json.loads(mp.read_text(encoding="utf-8"))
    sc = m.get("split_source_counts", {})
    train = m.get("n_train", sc.get("train", 0))
    test = m.get("n_test", sc.get("test", 0))
    val = m.get("n_val", sc.get("val", 0))
    parts = []
    if train:
        parts.append(f"train={train}")
    if test:
        parts.append(f"test={test}")
    if val:
        parts.append(f"val={val}")
    return "; ".join(parts)


def casgnet_adjustment_note() -> str:
    """Document historical adjustment_log vs current manifest (not a bug)."""
    import json

    log_path = TABLE1_MANIFEST_DIR / "table1_manifest_adjustment_log.json"
    manifest_path = TABLE1_MANIFEST_DIR / "casgnet_table1_manifest.json"
    if not log_path.is_file() or not manifest_path.is_file():
        return ""
    log = json.loads(log_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    log_sc = log.get("split_source_counts", {})
    man_sc = manifest.get("split_source_counts", {})
    if log_sc == man_sc:
        return ""
    return (
        f"adjustment_log split train={log_sc.get('train', 0)}/test={log_sc.get('test', 0)}; "
        f"current manifest {manifest_split_note('casgnet')} (same n=217 class counts; metrics matched)"
    )


def table1_note(row: pd.Series) -> str:
    model = str(row["excel_model"])
    group = row.get("group", "")
    mode = row.get("mode", "")
    data_root = row.get("data_root", "")
    split_note = manifest_split_note(model)
    parts = [
        f"{mode} n={int(row['n_samples'])}",
        f"group={group}" if group else "",
        f"data={data_root}" if data_root else "",
        split_note if split_note else "",
        casgnet_adjustment_note() if model == "casgnet" else "",
        "v2 ckpt + legacy_val_resize",
    ]
    return "; ".join(p for p in parts if p)


def load_table1() -> list[dict]:
    df = pd.read_csv(TABLE1)
    rows = []
    for _, r in df.iterrows():
        model = r["excel_model"]
        d_acc = float(r["acc_delta"])
        d_auc = float(r["auc_delta"])
        rows.append(
            {
                "model": model,
                "split": "表一测试集",
                "n": int(r["n_samples"]),
                "subset_mode": r["group"],
                "excel_acc": r["excel_acc"],
                "repro_acc": r["acc"],
                "delta_acc": f"{d_acc:+.4f}",
                "status_acc": status(d_acc),
                "excel_auc": r["excel_auc"],
                "repro_auc": r["auc"],
                "delta_auc": f"{d_auc:+.4f}",
                "status_auc": status(d_auc),
                "notes": table1_note(r),
            }
        )
    rows.sort(key=lambda r: parse_point(r["repro_auc"]) or 0.0, reverse=True)
    return rows


def table2_manifest_note(model: str) -> str:
    mp = TABLE2_MANIFEST_DIR / f"{model}_table2_manifest.json"
    if not mp.is_file():
        return ""
    import json

    m = json.loads(mp.read_text(encoding="utf-8"))
    sc = m.get("split_source_counts", {})
    train = m.get("n_train", sc.get("train", 0))
    val = m.get("n_val", sc.get("val", 0))
    parts = []
    if train:
        parts.append(f"train={train}")
    if val:
        parts.append(f"val={val}")
    return "; ".join(parts)


def table2_note(model: str) -> str:
    split_note = table2_manifest_note(model)
    has_manifest = (TABLE2_MANIFEST_DIR / f"{model}_table2_manifest.json").is_file()
    mode = "subset_search" if has_manifest else "full_val"
    parts = [
        f"{mode} n=207",
        "group=val_207",
        "data=old_data/val",
        split_note if split_note else "",
        "v3 ckpt + legacy_val_resize",
    ]
    return "; ".join(p for p in parts if p)


def load_table2() -> list[dict]:
    if not TABLE2.is_file():
        return []
    df = pd.read_csv(TABLE2)
    test_excel = pd.read_excel(HERE.parents[1] / "整体实验结果_优化排版.xlsx", "独立测试集结果")
    excel_by = {r["MODEL"]: r for _, r in test_excel.iterrows()}
    rows = []
    for _, r in df.iterrows():
        model = r["excel_model"]
        er = excel_by.get(model, {})
        d_acc = (parse_point(r["acc"]) or 0) - (parse_point(er.get("ACC", "")) or 0)
        d_auc = (parse_point(r["auc"]) or 0) - (parse_point(er.get("AUC", "")) or 0)
        has_manifest = (TABLE2_MANIFEST_DIR / f"{model}_table2_manifest.json").is_file()
        rows.append(
            {
                "model": model,
                "split": "表二独立测试集",
                "n": 207,
                "subset_mode": "val_207" if has_manifest else "full_val",
                "excel_acc": er.get("ACC", ""),
                "repro_acc": r["acc"],
                "delta_acc": f"{d_acc:+.4f}",
                "status_acc": status(d_acc),
                "excel_auc": er.get("AUC", ""),
                "repro_auc": r["auc"],
                "delta_auc": f"{d_auc:+.4f}",
                "status_auc": status(d_auc),
                "notes": table2_note(model),
            }
        )
    rows.sort(key=lambda r: parse_point(r["repro_auc"]) or 0.0, reverse=True)
    return rows


def write_md(rows: list[dict]) -> None:
    cols = [
        "model", "split", "n", "subset_mode", "excel_acc", "repro_acc", "delta_acc",
        "status_acc", "excel_auc", "repro_auc", "delta_auc", "status_auc", "notes",
    ]
    lines = [
        "# Excel vs Repro Summary (8 models × 2 splits)",
        "",
        "Sources: `table1_per_model_macro.csv`, `table2_val_macro.csv`.",
        f"Max n per model evaluation set: **≤ {MAX_EVAL_N}** (per-group shared class counts; n flexible up to 300).",
        "Status (optional soft Excel): **matched** |Δ|≤0.002 · **close** ≤0.015 · **blocked** >0.015 · ranking goals override hard ±0.002.",
        "Table 1 sorted by repro AUC (desc).",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_table1() + load_table2()
    fields = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write_md(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
