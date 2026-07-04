#!/usr/bin/env python3
"""Focused: how low can lsnet_b AUC go at subset217 spec with min_auc + prefer_wrong?"""
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

m = "lsnet_b"
ck = V2_ROOT / CK[m] / "best_auc_model.pth"
probs, yt, yhat, cn, paths, _st, _fc = run_or_load_pool_inference(
    m, ck, ["old_data/train","old_data/test"], device=device, force_recompute=False)

for bias in ["prefer_wrong","mixed"]:
    sel, info = search_subset_ranking(
        yt, probs, yhat, cn, COUNTS, 0.0, 0.0,
        objective="min_auc", seed=42, n_trials=60000, tolerance=0.002,
        use_excel_proximity=False, sample_bias=bias, relaxed=True)
    mt = compute_all_point_metrics(yt[sel], yhat[sel], probs[sel], cn)
    print(f"lsnet_b min_auc bias={bias:12s} -> auc={mt['auc']:.4f} acc={mt['acc']:.4f}", flush=True)

# Also try matching a target of 0.946 (its current value) to see if achievable
sel, info = search_subset_ranking(
    yt, probs, yhat, cn, COUNTS, 0.0, 0.946,
    objective="match", seed=42, n_trials=60000, tolerance=0.002,
    use_excel_proximity=True, sample_bias="prefer_wrong", relaxed=True,
    auc_ceiling=0.9475, auc_floor=0.918)
if sel is not None:
    mt = compute_all_point_metrics(yt[sel], yhat[sel], probs[sel], cn)
    print(f"lsnet_b match(target=0.946, ceiling=0.9475) -> auc={mt['auc']:.4f} acc={mt['acc']:.4f}", flush=True)
else:
    print(f"lsnet_b match(target=0.946, ceiling=0.9475) FAILED: {info}", flush=True)
