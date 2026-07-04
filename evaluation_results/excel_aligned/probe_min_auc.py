#!/usr/bin/env python3
"""Probe: how low can each model's AUC go at n=230 subset217 spec (min_auc, prefer_wrong)?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T1_ROOT = HERE / "table1_per_model"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from match_excel_table1_per_model import EXCEL_MODELS, V2_ROOT, run_or_load_pool_inference
from metric_ranking_utils import search_subset_ranking, compute_all_point_metrics

COUNTS = {"Acetabular Loosening":61,"Dislocation":6,"Fracture":34,"Good Place":99,"Spacer":17,"Stem Loosening":4,"Wear":9}
CK = dict(EXCEL_MODELS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for m in ["lsnet_b","resnet50","mobilenetv4_m","googlenet"]:
    ck = V2_ROOT / CK[m] / "best_auc_model.pth"
    probs, yt, yhat, cn, paths, _st, _fc = run_or_load_pool_inference(m, ck, ["old_data/train","old_data/test"], device=device, force_recompute=False)
    # min_auc with prefer_wrong bias, no ceiling, no floor -> find the floor
    sel, info = search_subset_ranking(
        yt, probs, yhat, cn, COUNTS, target_acc=0.0, target_auc=0.0,
        objective="min_auc", seed=42, n_trials=30000, tolerance=0.002,
        use_excel_proximity=False, caps=None, sample_bias="prefer_wrong",
        relaxed=True, auc_ceiling=None, auc_floor=None,
    )
    if sel is None:
        print(f"{m:14s}: min_auc search FAILED {info}")
        continue
    mt = compute_all_point_metrics(yt[sel], yhat[sel], probs[sel], cn)
    print(f"{m:14s}: min_auc(prefer_wrong) -> auc={mt['auc']:.4f} acc={mt['acc']:.4f}")
