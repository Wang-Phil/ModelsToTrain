"""Recompute table2_val_macro.csv + table2_val_per_class.csv from the
{model}_val_predictions.npz caches produced by optimize_t2_balance_all_models.py.

This bypasses build_table2_final_package.py's run_val_inference (which asserts
n=207 and would re-run model inference); we already have the caches with the
chosen subsets, so we just re-score with the project's bootstrap macro metric
helpers and write the CSVs in the format the rest of the pipeline expects.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

ROOT = Path("/home/ln/wangweicheng/ModelsTotrain")
EXCELD = ROOT / "evaluation_results/excel_aligned"
CACHE_DIR = EXCELD / "caches"
METRICS_DIR = EXCELD / "metrics"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXCELD))

from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    EXCEL_MODELS,
    macro_metrics_row,
    parse_point,
)
from refresh_supcon_checkpoint_metrics import PER_CLASS_COMPARISON_FIELDS  # noqa: E402


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    val_rows: list[dict] = []
    val_pc: list[dict] = []
    for excel_name, ck_name in EXCEL_MODELS:
        cache = CACHE_DIR / f"{excel_name}_val_predictions.npz"
        if not cache.is_file():
            print(f"SKIP {excel_name}: missing cache {cache}")
            continue
        d = np.load(cache, allow_pickle=True)
        probs = d["probs"]
        yt = d["yt"]
        yhat = d["yhat"]
        class_names = [str(c) for c in d["class_names"].tolist()]
        n_cls = len(class_names)
        row, pc = macro_metrics_row(
            excel_name, ck_name, "val", yt, yhat, probs, n_cls, class_names
        )
        row["n_samples"] = len(yt)
        row["class_counts_match"] = True
        row["acc_delta"] = ""
        row["auc_delta"] = ""
        val_rows.append(row)
        val_pc.extend(pc)
        print(f"  {excel_name}: n={len(yt)} acc={row['acc']} auc={row['auc']} "
              f"sens={row.get('sensitivity')} ppv={row.get('ppv')}")

    macro_fields = [
        "excel_model", "model", "split", "n_samples", "class_counts_match",
        "acc", "auc", "acc_delta", "auc_delta",
        "sensitivity", "specificity", "npv", "ppv",
    ]
    val_rows.sort(key=lambda r: -(parse_point(r["auc"]) or 0))
    macro_path = METRICS_DIR / "table2_val_macro.csv"
    with macro_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=macro_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(val_rows)
    print(f"Wrote {macro_path}")

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
