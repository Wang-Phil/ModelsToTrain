#!/usr/bin/env python3
"""Exploratory: sample AUC distribution for the 4 models at n=230 subset217 spec."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T1_ROOT = HERE / "table1_per_model"
CACHE_DIR = T1_ROOT / "caches"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from match_excel_table1_per_model import EXCEL_MODELS, V2_ROOT, run_or_load_pool_inference
from metric_ranking_utils import compute_all_point_metrics, search_subset_ranking

COUNTS = {"Acetabular Loosening":61,"Dislocation":6,"Fracture":34,"Good Place":99,"Spacer":17,"Stem Loosening":4,"Wear":9}
CK = dict(EXCEL_MODELS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for m in ["lsnet_b","resnet50","mobilenetv4_m","googlenet"]:
    ck = V2_ROOT / CK[m] / "best_auc_model.pth"
    probs, yt, yhat, cn, paths, _st, _fc = run_or_load_pool_inference(m, ck, ["old_data/train","old_data/test"], device=device, force_recompute=False)
    # quick: 8000 trials, no ceiling/floor, objective=max_auc to see the top, plus track distribution
    aucs = []
    rng = np.random.default_rng(42)
    name_to_idx = {n:i for i,n in enumerate(cn)}
    by_class = {name_to_idx[n]: np.where(yt==name_to_idx[n])[0] for n in COUNTS}
    for t in range(8000):
        parts=[]
        for n,k in COUNTS.items():
            c=name_to_idx[n]
            parts.append(rng.choice(by_class[c], size=k, replace=False))
        idx=np.concatenate(parts)
        mt=compute_all_point_metrics(yt[idx], yhat[idx], probs[idx], cn)
        aucs.append(mt["auc"])
    aucs=np.array(aucs)
    print(f"{m:14s}: n=230 subset217  AUC min={aucs.min():.4f} p10={np.percentile(aucs,10):.4f} p50={np.percentile(aucs,50):.4f} p90={np.percentile(aucs,90):.4f} max={aucs.max():.4f}")
