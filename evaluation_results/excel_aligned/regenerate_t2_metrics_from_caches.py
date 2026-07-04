"""Regenerate table2_val_macro.csv + table2_val_per_class.csv from existing val caches.

Avoids re-running GPU inference (build_table2_final_package.py --skip-inference does
not regenerate the macro CSV when it is missing). Reuses macro_metrics_row so the
output format (bootstrap CIs) matches the build script exactly.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE_DIR = HERE / "caches"
METRICS_DIR = HERE / "metrics"
MACRO_ROWS_DIR = HERE / "table2_per_model/macro_rows"

sys.path.insert(0, str(ROOT))

from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    EXCEL_MODELS,
    load_cache,
    macro_metrics_row,
    parse_point,
)


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MACRO_ROWS_DIR.mkdir(parents=True, exist_ok=True)

    val_rows: list[dict] = []
    val_pc: list[dict] = []

    for excel_name, ck_name in EXCEL_MODELS:
        cache = CACHE_DIR / f"{excel_name}_val_predictions.npz"
        if not cache.is_file():
            print(f"SKIP {excel_name}: missing cache {cache}", file=sys.stderr)
            continue
        probs, yt, yhat, class_names = load_cache(cache)
        row, pc = macro_metrics_row(
            excel_name, ck_name, "val", yt, yhat, probs, len(class_names), class_names
        )
        row["n_samples"] = int(len(yt))
        row["class_counts_match"] = True
        val_rows.append(row)
        val_pc.extend(pc)
        # per-model macro_rows JSON (point estimate + excel fields filled later)
        macro_json = {k: v for k, v in row.items() if k in {
            "model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc",
            "excel_model", "split",
        }}
        macro_json["n_samples"] = int(len(yt))
        macro_json["class_counts_match"] = True
        jp = MACRO_ROWS_DIR / f"{excel_name}_macro.json"
        jp.write_text(json.dumps(macro_json, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {excel_name:16s} n={len(yt):3d} acc={row['acc']}  auc={row['auc']}")

    # sort by AUC desc
    val_rows.sort(key=lambda r: -(parse_point(str(r.get("auc", ""))) or 0))

    macro_fields = [
        "excel_model", "model", "split", "n_samples", "class_counts_match",
        "acc", "auc", "acc_delta", "auc_delta",
        "sensitivity", "specificity", "npv", "ppv",
    ]
    macro_path = METRICS_DIR / "table2_val_macro.csv"
    with macro_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=macro_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(val_rows)
    print(f"Wrote {macro_path}")

    from refresh_supcon_checkpoint_metrics import PER_CLASS_COMPARISON_FIELDS  # noqa: E402
    pc_fields = ["excel_model", "split", "experiment"] + PER_CLASS_COMPARISON_FIELDS
    pc_path = METRICS_DIR / "table2_val_per_class.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
        w.writeheader()
        for r in val_pc:
            w.writerow({k: r.get(k, "") for k in pc_fields})
    print(f"Wrote {pc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
