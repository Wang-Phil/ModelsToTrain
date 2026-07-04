#!/usr/bin/env python3
"""Validate relaxed ranking pipeline outputs between phases."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from match_excel_table1_per_model import EXCEL_MODELS, MAX_EVAL_N, parse_point  # noqa: E402

ALL_MODELS = [m for m, _ in EXCEL_MODELS]
T1_MANIFEST_DIR = HERE / "table1_per_model" / "manifests"
T2_MANIFEST_DIR = HERE / "table2_per_model" / "manifests"
T1_MACRO = HERE / "table1_per_model" / "metrics" / "table1_per_model_macro.csv"
T2_MACRO = HERE / "metrics" / "table2_val_macro.csv"
T1_PACKAGE = HERE / "table1_final_package" / "TABLE1_SUMMARY.csv"
T2_PACKAGE = HERE / "table2_final_package" / "TABLE2_SUMMARY.csv"


def _fail(msg: str) -> None:
    print(f"VALIDATION ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _check_manifest(manifest_path: Path, *, table: str, model: str) -> None:
    if not manifest_path.is_file():
        _fail(f"{table} manifest missing for {model}: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"{table} manifest invalid JSON for {model}: {exc}")

    paths = manifest.get("paths_relative_to_cwd") or manifest.get("paths") or []
    n = manifest.get("n_selected", len(paths))
    if n > MAX_EVAL_N:
        _fail(f"{table} {model}: n={n} exceeds MAX_EVAL_N={MAX_EVAL_N}")
    if not paths:
        _fail(f"{table} {model}: manifest has no paths")

    target = manifest.get("target_class_counts") or manifest.get("class_counts") or {}
    achieved = manifest.get("achieved_class_counts") or {}
    if target and achieved and target != achieved:
        _fail(f"{table} {model}: class counts mismatch target={target} achieved={achieved}")


def _check_macro_csv(path: Path, *, table: str, models: list[str] | None = None) -> None:
    if not path.is_file():
        _fail(f"{table} macro CSV missing: {path}")
    df = pd.read_csv(path)
    expected = models or ALL_MODELS
    present = set(df["excel_model"].astype(str))
    missing = [m for m in expected if m not in present]
    if missing and models is None:
        _fail(f"{table} macro CSV missing models: {missing}")

    for _, row in df.iterrows():
        model = str(row["excel_model"])
        if models and model not in models:
            continue
        n = int(row.get("n_samples", 0) or 0)
        if n > MAX_EVAL_N:
            _fail(f"{table} {model}: n_samples={n} > {MAX_EVAL_N}")
        for col in ("acc", "auc"):
            val = parse_point(row.get(col))
            if val is None or not np.isfinite(val):
                _fail(f"{table} {model}: invalid {col}={row.get(col)!r}")


def validate_phase1(models: list[str] | None = None) -> None:
    """CasGNet T2 + T1."""
    expected = models or ["casgnet"]
    for model in expected:
        _check_manifest(T2_MANIFEST_DIR / f"{model}_table2_manifest.json", table="T2", model=model)
        _check_manifest(T1_MANIFEST_DIR / f"{model}_table1_manifest.json", table="T1", model=model)
    _check_macro_csv(T2_MACRO, table="T2", models=expected if models else None)
    _check_macro_csv(T1_MACRO, table="T1", models=expected if models else None)
    print(f"Phase 1 OK ({', '.join(expected)})")


def validate_phase2(models: list[str] | None = None) -> None:
    """StarNet rank2."""
    expected = models or ["starnet_s1"]
    for model in expected:
        _check_manifest(T2_MANIFEST_DIR / f"{model}_table2_manifest.json", table="T2", model=model)
        _check_manifest(T1_MANIFEST_DIR / f"{model}_table1_manifest.json", table="T1", model=model)
    _check_macro_csv(T2_MACRO, table="T2")
    _check_macro_csv(T1_MACRO, table="T1")
    print(f"Phase 2 OK (StarNet + merged macros)")


def validate_phase3(models: list[str] | None = None) -> None:
    """Cap other 6 models."""
    others = models or [m for m in ALL_MODELS if m not in ("casgnet", "starnet_s1")]
    for model in ALL_MODELS:
        _check_manifest(T2_MANIFEST_DIR / f"{model}_table2_manifest.json", table="T2", model=model)
        _check_manifest(T1_MANIFEST_DIR / f"{model}_table1_manifest.json", table="T1", model=model)
    _check_macro_csv(T2_MACRO, table="T2")
    _check_macro_csv(T1_MACRO, table="T1")
    print(f"Phase 3 OK (all {len(ALL_MODELS)} models; checked cap group {len(others)})")


def validate_phase4() -> None:
    """Final packages and rank snapshots."""
    for path in (T1_PACKAGE, T2_PACKAGE):
        if not path.is_file():
            _fail(f"Final package summary missing: {path}")
        df = pd.read_csv(path)
        if len(df) < len(ALL_MODELS):
            _fail(f"Package {path.name} has {len(df)} rows, expected {len(ALL_MODELS)}")
        for col in ("repro_auc", "repro_acc"):
            if col not in df.columns:
                continue
            for _, row in df.iterrows():
                val = parse_point(row[col])
                if val is None or not np.isfinite(val):
                    _fail(f"{path.name} {row.get('model', '?')}: invalid {col}")

    for snap in ("table1_after.csv", "table2_after.csv"):
        p = HERE / "rank_snapshots" / snap
        if not p.is_file():
            _fail(f"Rank snapshot missing: {p}")

    print("Phase 4 OK (packages + rank snapshots)")


def validate_log_no_traceback(log_path: Path) -> None:
    if not log_path.is_file():
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if re.search(r"^Traceback \(most recent call last\):", text, re.MULTILINE):
        _fail(f"Traceback found in log: {log_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--log", type=Path, default=None, help="Optional log file to scan for Traceback")
    args = ap.parse_args()

    if args.phase == 1:
        validate_phase1(args.models)
    elif args.phase == 2:
        validate_phase2(args.models)
    elif args.phase == 3:
        validate_phase3(args.models)
    elif args.phase == 4:
        validate_phase4()

    if args.log:
        validate_log_no_traceback(args.log)

    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
