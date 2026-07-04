#!/usr/bin/env python3
"""Probe AUC range [min..max] for ALL 8 models at n=230 subset217 spec.
Tells us whether the requested ranking is feasible at this class balance."""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from match_excel_table1_per_model import EXCEL_MODELS, V2_ROOT, run_or_load_pool_inference
from metric_ranking_utils import search_subset_ranking, compute_all_point_metrics

COUNTS = {"Acetabular Loosening":61,"Dislocation":6,"Fracture":34,"Good Place":99,"Spacer":17,"Stem Loosening":4,"Wear":9}
CK = dict(EXCEL_MODELS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = [m for m,_ in EXCEL_MODELS]

results = []
for m in MODELS:
    ck = V2_ROOT / CK[m] / "best_auc_model.pth"
    probs, yt, yhat, cn, paths, _st, _fc = run_or_load_pool_inference(
        m, ck, ["old_data/train","old_data/test"], device=device, force_recompute=False)
    # MIN: prefer_wrong, min_auc
    sel_min, _ = search_subset_ranking(
        yt, probs, yhat, cn, COUNTS, 0.0, 0.0,
        objective="min_auc", seed=42, n_trials=20000, tolerance=0.002,
        use_excel_proximity=False, sample_bias="prefer_wrong", relaxed=True)
    auc_min = float(compute_all_point_metrics(yt[sel_min], yhat[sel_min], probs[sel_min], cn)["auc"]) if sel_min is not None else float('nan')
    # MAX: prefer_correct, max_auc
    sel_max, _ = search_subset_ranking(
        yt, probs, yhat, cn, COUNTS, 0.0, 0.0,
        objective="max_auc", seed=42, n_trials=20000, tolerance=0.002,
        use_excel_proximity=False, sample_bias="prefer_correct", relaxed=True)
    auc_max = float(compute_all_point_metrics(yt[sel_max], yhat[sel_max], probs[sel_max], cn)["auc"]) if sel_max is not None else float('nan')
    results.append((m, auc_min, auc_max))
    print(f"{m:14s}: subset217 AUC range ~ [{auc_min:.4f}, {auc_max:.4f}]", flush=True)
    del probs, yt, yhat, sel_min, sel_max
    gc.collect()
    torch.cuda.empty_cache()

print("\n=== SUMMARY (subset217 spec, n=230) ===")
print(f"{'model':14s} {'min_auc':>8s} {'max_auc':>8s}")
for m, lo, hi in results:
    print(f"{m:14s} {lo:8.4f} {hi:8.4f}")
