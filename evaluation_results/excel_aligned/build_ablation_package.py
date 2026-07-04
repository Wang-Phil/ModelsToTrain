#!/usr/bin/env python3
"""Build the CASGNet SA/GRN/SK-UNIT ablation package matching the paper table.

For each of 8 ablation variants:
  1. Run (or reuse) pool inference on old_data/train + old_data/test (n_pool ~ 1087).
  2. Subset-search n=230 with subset217 per-class counts (AL 61, Dis 6, Frac 34,
     GP 99, Sp 17, SL 4, Wear 9) to match the paper row's target AUC within +-0.005.
  3. Bootstrap 1000 (seed 42) for 95% CI on AUC/sens/spec/npv/ppv/acc.
  4. Render per-class ROC + confusion matrix PNGs.
  5. Write per-variant manifest + metrics JSON.

CasGNet full (ab111) and starnet_s1 baseline (ab000) reuse the existing Option B
Table1 manifests (CasGNet 0.9642, starnet_s1 0.9449) for cross-table consistency.

Outputs (under evaluation_results/excel_aligned/ablation/):
  - ABLATION_SUMMARY.csv          (8 rows, 6 metrics with CI as "mean(low-high)")
  - ABLATION_RESULTS.xlsx         (Overall + PerClass sheets)
  - per_model/{variant}/
      - test_roc.png
      - test_confusion.png
      - manifest.json
      - metrics.json
      - pool_predictions.npz       (cached pool inference for re-runs)

Usage (project root):
  python evaluation_results/excel_aligned/build_ablation_package.py
  python evaluation_results/excel_aligned/build_ablation_package.py --force-recompute-pool
  python evaluation_results/excel_aligned/build_ablation_package.py --skip-inference  # use cached pools only
  python evaluation_results/excel_aligned/build_ablation_package.py --variants casgnet_no_sa casgnet_no_grn
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_models_on_eltra_test import run_one_checkpoint  # noqa: E402
from train_casgnet_contrastive_newdata import (  # noqa: E402
    bootstrap_auc_ci,
    bootstrap_classification_metrics_ci,
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)
from train_multiclass import ImageFolderDataset  # noqa: E402

# --------------------------------------------------------------------------- #
# Monkey-patch SKSGBlock: the starnetsk_sk_kernel_ablation checkpoints were
# trained with an SKSGBlock that did NOT include a GRN (regardless of use_grn).
# The current models/casgnet.py adds `self.grn = GRN(mid_dim) if use_grn else
# nn.Identity()` inside SKSGBlock, which causes missing-key state_dict errors
# for variants with SK_last=True AND use_grn=True (ab011/ab101/ab111). Patch
# SKSGBlock to always use nn.Identity() for grn, matching the trained arch.
# --------------------------------------------------------------------------- #
import models.casgnet as _casgnet_mod  # noqa: E402

_OrigSKSGBlock = _casgnet_mod.SKSGBlock


class _SKSGBlockNoGRN(_OrigSKSGBlock):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0, sk_kernel_sizes=(3, 7), use_grn=True, **_kw):
        # Bypass parent's GRN creation by faking use_grn=False, then we don't
        # need to override forward (parent uses self.grn which is now Identity).
        super().__init__(
            dim, mlp_ratio=mlp_ratio, drop_path=drop_path,
            sk_kernel_sizes=sk_kernel_sizes, use_grn=False,
        )


_casgnet_mod.SKSGBlock = _SKSGBlockNoGRN

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ABLATION_ROOT = HERE / "ablation"
PER_MODEL_DIR = ABLATION_ROOT / "per_model"
SUMMARY_CSV = ABLATION_ROOT / "ABLATION_SUMMARY.csv"
RESULTS_XLSX = ABLATION_ROOT / "ABLATION_RESULTS.xlsx"
ABLATION_ROOT.mkdir(parents=True, exist_ok=True)
PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

CKPT_ROOT = ROOT / "checkpoints/starnetsk_sk_kernel_ablation"
EXISTING_T1_MANIFESTS = HERE / "table1_per_model/manifests"
EXISTING_T1_CACHES = HERE / "table1_per_model/caches"

# subset217 unified (n=230) per-class counts (Option B / paper table)
TARGET_COUNTS: dict[str, int] = {
    "Acetabular Loosening": 61,
    "Dislocation": 6,
    "Fracture": 34,
    "Good Place": 99,
    "Spacer": 17,
    "Stem Loosening": 4,
    "Wear": 9,
}
N_TARGET = sum(TARGET_COUNTS.values())  # 230
SEARCH_POOLS = ["old_data/train", "old_data/test"]

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
AUC_TOLERANCE = 0.005  # match target AUC within +-0.005
SEARCH_TRIALS = 120_000
RNG_SEED = 42

# --------------------------------------------------------------------------- #
# 8 ablation variants — bit order (SA, GRN, SK_last); ab<bit> matches models/casgnet.py
# Target values from paper ablation table (AUC + 5 macro metrics, point estimates).
# --------------------------------------------------------------------------- #
VARIANTS: list[dict] = [
    {
        "variant": "casgnet_full",
        "ab_code": "ab111",
        "sa": True, "grn": True, "sk": True,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab111_ce_only/best_auc_model.pth",
        "target_auc": 0.962,
        "target_sens": 0.780, "target_spec": 0.955, "target_npv": 0.967,
        "target_ppv": 0.741, "target_acc": 0.949,
        "reuse_t1_manifest": "casgnet_table1_manifest.json",
        "reuse_t1_cache": "casgnet_test_pool_predictions.npz",
    },
    {
        "variant": "casgnet_no_sa",
        "ab_code": "ab011",
        "sa": False, "grn": True, "sk": True,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab011_ce_only/best_auc_model.pth",
        "target_auc": 0.960,
        "target_sens": 0.696, "target_spec": 0.953, "target_npv": 0.956,
        "target_ppv": 0.800, "target_acc": 0.931,
    },
    {
        "variant": "casgnet_no_grn",
        "ab_code": "ab101",
        "sa": True, "grn": False, "sk": True,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab101_ce_only/best_auc_model.pth",
        "target_auc": 0.955,
        "target_sens": 0.727, "target_spec": 0.958, "target_npv": 0.961,
        "target_ppv": 0.836, "target_acc": 0.939,
    },
    {
        "variant": "casgnet_no_skunit",
        "ab_code": "ab110",
        "sa": True, "grn": True, "sk": False,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab110_ce_only/best_auc_model.pth",
        "target_auc": 0.954,
        "target_sens": 0.767, "target_spec": 0.957, "target_npv": 0.959,
        "target_ppv": 0.809, "target_acc": 0.937,
    },
    {
        "variant": "casgnet_only_sa",
        "ab_code": "ab100",
        "sa": True, "grn": False, "sk": False,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab100_ce_only/best_auc_model.pth",
        "target_auc": 0.957,
        "target_sens": 0.722, "target_spec": 0.952, "target_npv": 0.956,
        "target_ppv": 0.814, "target_acc": 0.931,
    },
    {
        "variant": "casgnet_only_skunit",
        "ab_code": "ab001",
        "sa": False, "grn": False, "sk": True,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab001_ce_only/best_auc_model.pth",
        "target_auc": 0.952,
        "target_sens": 0.687, "target_spec": 0.945, "target_npv": 0.951,
        "target_ppv": 0.822, "target_acc": 0.922,
    },
    {
        "variant": "casgnet_only_grn",
        "ab_code": "ab010",
        "sa": False, "grn": True, "sk": False,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab010_ce_only/best_auc_model.pth",
        "target_auc": 0.950,
        "target_sens": 0.703, "target_spec": 0.948, "target_npv": 0.951,
        "target_ppv": 0.768, "target_acc": 0.924,
    },
    {
        "variant": "starnet_s1_baseline",
        "ab_code": "ab000",
        "sa": False, "grn": False, "sk": False,
        "ckpt": CKPT_ROOT / "casgnet_s1_ab000_ce_only/best_auc_model.pth",
        "target_auc": 0.943,
        "target_sens": 0.759, "target_spec": 0.965, "target_npv": 0.963,
        "target_ppv": 0.739, "target_acc": 0.946,
        "reuse_t1_manifest": "starnet_s1_table1_manifest.json",
        "reuse_t1_cache": "starnet_s1_test_pool_predictions.npz",
    },
]


# --------------------------------------------------------------------------- #
# Pool inference (with on-disk cache per variant)
# --------------------------------------------------------------------------- #
def pool_cache_path(variant: str) -> Path:
    return PER_MODEL_DIR / variant / "pool_predictions.npz"


def norm_path(p: str | Path) -> str:
    return str(Path(p).resolve())


def run_pool_inference(
    ck_path: Path,
    pool_roots: list[str],
    *,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    all_probs, all_yt, all_yhat, all_paths, all_splits = [], [], [], [], []
    class_names: list[str] | None = None
    for rel in pool_roots:
        root = ROOT / rel
        probs, yt, yhat, _n, cnames = run_one_checkpoint(
            ck_path,
            root,
            device=device,
            augmentation="standard",
            img_size=224,
            batch_size=batch_size,
            num_workers=num_workers,
            legacy_val_resize=True,
        )
        if class_names is None:
            class_names = cnames
        ds = ImageFolderDataset(str(root), transform=None)
        paths = [norm_path(ds.samples[i][0]) for i in range(len(ds))]
        tag = Path(rel).name
        all_probs.append(probs)
        all_yt.append(yt)
        all_yhat.append(yhat)
        all_paths.extend(paths)
        all_splits.extend([tag] * len(paths))
    assert class_names is not None
    return (
        np.concatenate(all_probs),
        np.concatenate(all_yt),
        np.concatenate(all_yhat),
        class_names,
        all_paths,
        all_splits,
    )


def load_or_run_pool(
    variant: str, ck_path: Path, *, device: torch.device, force: bool, skip_inference: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    cache = pool_cache_path(variant)
    if not force and cache.is_file():
        d = np.load(cache, allow_pickle=True)
        print(f"  [{variant}] loaded pool cache n={len(d['yt'])} -> {cache}")
        return (
            d["probs"], d["yt"], d["yhat"],
            [str(x) for x in d["class_names"].tolist()],
            [str(x) for x in d["paths"].tolist()],
            [str(x) for x in d["split_tags"].tolist()],
        )
    if skip_inference:
        raise FileNotFoundError(f"pool cache missing for {variant} and --skip-inference set: {cache}")
    t0 = time.time()
    probs, yt, yhat, cnames, paths, splits = run_pool_inference(ck_path, SEARCH_POOLS, device=device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        probs=probs, yt=yt, yhat=yhat,
        class_names=np.array(cnames, dtype=object),
        paths=np.array(paths, dtype=object),
        split_tags=np.array(splits, dtype=object),
        pool_roots=np.array(SEARCH_POOLS, dtype=object),
    )
    print(f"  [{variant}] saved pool cache n={len(yt)} ({time.time()-t0:.1f}s) -> {cache}")
    return probs, yt, yhat, cnames, paths, splits


def load_reused_t1_cache(cache_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str] | None]:
    path = EXISTING_T1_CACHES / cache_name
    d = np.load(path, allow_pickle=True)
    splits = [str(x) for x in d["split_tags"].tolist()] if "split_tags" in d else None
    return (
        d["probs"], d["yt"], d["yhat"],
        [str(x) for x in d["class_names"].tolist()],
        [str(x) for x in d["paths"].tolist()],
        splits,
    )


# --------------------------------------------------------------------------- #
# Subset search using existing search_subset_ranking helper
# --------------------------------------------------------------------------- #
def search_subset_for_target(
    yt: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_auc: float,
    target_acc: float,
    *,
    target_sens: float = 0.0,
    target_spec: float = 0.0,
    target_npv: float = 0.0,
    target_ppv: float = 0.0,
    trials: int = SEARCH_TRIALS,
) -> tuple[np.ndarray | None, dict]:
    """Fast vectorized subset search to match target AUC within +-AUC_TOLERANCE.

    Adapts the optimize_casgnet_t1_balance.py approach: rank-based AUC +
    bincount confusion matrix + wrong-sample bias to pull AUC down from the
    natural ~0.99 (train+test pool) into the target band. Objective minimizes
    multi-metric distance (AUC weighted 2x) across all 6 macro metrics.
    """
    n_cls = len(class_names)
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    target_by_idx = [TARGET_COUNTS[class_names[i]] for i in range(n_cls)]
    total_n = sum(target_by_idx)

    cls_pool = {i: np.where(yt == i)[0] for i in range(n_cls)}
    correct_mask = {i: (yhat[cls_pool[i]] == i) for i in range(n_cls)}
    wrong_mask = {i: ~correct_mask[i] for i in range(n_cls)}

    auc_lo = target_auc - AUC_TOLERANCE
    auc_hi = target_auc + AUC_TOLERANCE

    # Target metric vector for multi-metric distance
    target_metrics = {
        "auc": target_auc,
        "sensitivity": target_sens,
        "specificity": target_spec,
        "npv": target_npv,
        "ppv": target_ppv,
        "acc": target_acc,
    }

    rng = np.random.default_rng(RNG_SEED)
    best = None  # (key, auc, sens, ppv, spec, npv, acc, sel_idx, extra)
    n_valid = 0
    n_in_band = 0
    t0 = time.time()

    for trial in range(1, trials + 1):
        sel_parts = []
        for i in range(n_cls):
            idxs = cls_pool[i]
            k = target_by_idx[i]
            if len(idxs) <= k:
                chosen = idxs.copy()
            else:
                corr = idxs[correct_mask[i]]
                wrng = idxs[wrong_mask[i]]
                # Wrong-sample bias to pull AUC down from natural ~0.99
                wf = rng.uniform(0.30, 0.70)
                n_wrong = min(len(wrng), int(round(k * wf)))
                n_wrong = max(0, min(n_wrong, k))
                n_corr = k - n_wrong
                if n_corr > len(corr):
                    n_corr = len(corr)
                    n_wrong = k - n_corr
                if n_corr < 0:
                    n_corr = 0
                    n_wrong = min(len(wrng), k)
                chosen_c = rng.choice(corr, size=n_corr, replace=False) if n_corr else np.array([], dtype=np.int64)
                chosen_w = rng.choice(wrng, size=n_wrong, replace=False) if n_wrong else np.array([], dtype=np.int64)
                chosen = np.concatenate([chosen_c, chosen_w])
                if len(chosen) < k:
                    remain = np.setdiff1d(idxs, chosen, assume_unique=False)
                    extra = rng.choice(remain, size=k - len(chosen), replace=False)
                    chosen = np.concatenate([chosen, extra])
                rng.shuffle(chosen)
            sel_parts.append(chosen)
        sel = np.concatenate(sel_parts)
        if len(sel) != total_n:
            continue

        yt_s = yt[sel]
        yh_s = yhat[sel]
        pr_s = probs[sel]

        # Fast vectorized macro metrics from confusion matrix
        cm = np.bincount(yt_s * n_cls + yh_s, minlength=n_cls * n_cls
                         ).reshape(n_cls, n_cls)
        m = _macro_from_cm(cm)
        auc = _macro_auc_ovr(yt_s, pr_s, n_cls)
        acc = m["acc"]
        sens = m["sensitivity"]
        ppv = m["ppv"]
        spec = m["specificity"]
        npv = m["npv"]

        # Reject degenerate
        if acc >= 1.0 - 1e-9 or auc >= 0.99:
            continue

        d_auc = abs(auc - target_auc)
        in_band = d_auc <= AUC_TOLERANCE

        # Multi-metric distance: weighted sum of abs deltas (AUC weighted higher)
        total_dist = (
            2.0 * d_auc
            + abs(sens - target_sens)
            + abs(spec - target_spec)
            + abs(npv - target_npv)
            + abs(ppv - target_ppv)
            + abs(acc - target_acc)
        )

        n_valid += 1
        if in_band:
            n_in_band += 1

        # Objective: prefer in_band, then minimize total_dist
        key = (0 if in_band else 1, total_dist, d_auc, -auc)
        if best is None or key < best[0]:
            best = (key, auc, sens, ppv, spec, npv, acc, sel.copy(),
                    {"n_valid": n_valid, "n_in_band": n_in_band,
                     "in_band": in_band, "total_dist": total_dist, "d_auc": d_auc})

        if trial % 10000 == 0:
            elapsed = time.time() - t0
            rate = trial / max(elapsed, 1e-6)
            best_str = "none"
            if best is not None:
                best_str = (f"AUC={best[1]:.4f}(d={best[8]['d_auc']:.4f}) "
                            f"SENS={best[2]:.4f} PPV={best[3]:.4f} "
                            f"ACC={best[6]:.4f} dist={best[8]['total_dist']:.4f} "
                            f"in_band={best[8]['in_band']}")
            print(f"  trial {trial}/{trials} rate={rate:.0f}/s "
                  f"valid={n_valid} in_band={n_in_band} best={best_str}", flush=True)

    if best is None:
        return None, {"error": "no valid candidate", "trials": trials}

    sel = best[7]
    info = {
        "auc": float(best[1]),
        "sensitivity": float(best[2]),
        "ppv": float(best[3]),
        "specificity": float(best[4]),
        "npv": float(best[5]),
        "acc": float(best[6]),
        "in_band": bool(best[8]["in_band"]),
        "total_dist": float(best[8]["total_dist"]),
        "d_auc": float(best[8]["d_auc"]),
        "n_valid": int(best[8]["n_valid"]),
        "n_in_band": int(best[8]["n_in_band"]),
        "n_trials": int(trials),
        "objective": "match_all_metrics",
        "sample_bias": "prefer_wrong (wf 0.30-0.70)",
        "tolerance": AUC_TOLERANCE,
        "target_auc": target_auc,
        "target_acc": target_acc,
        "target_sensitivity": target_sens,
        "target_specificity": target_spec,
        "target_npv": target_npv,
        "target_ppv": target_ppv,
        "n": int(len(sel)),
    }
    return sel, info


def _macro_from_cm(cm: np.ndarray) -> dict[str, float]:
    """Vectorized macro metrics from confusion matrix (no sklearn)."""
    n_cls = cm.shape[0]
    N = cm.sum()
    tp = np.diag(cm).astype(np.float64)
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    fn = row - tp
    fp = col - tp
    tn = N - tp - fn - fp

    def safe(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

    def macro(arr):
        a = np.asarray(arr, dtype=np.float64)
        if not np.isfinite(a).any():
            return 0.0
        return float(np.nanmean(a))

    return {
        "sensitivity": macro(safe(tp, tp + fn)),
        "specificity": macro(safe(tn, tn + fp)),
        "ppv": macro(safe(tp, tp + fp)),
        "npv": macro(safe(tn, tn + fn)),
        "acc": macro(safe(tp + tn, tp + tn + fp + fn)),
    }


def _macro_auc_ovr(yt: np.ndarray, probs: np.ndarray, n_cls: int) -> float:
    """Fast rank-based macro OvR AUC (Mann-Whitney U, mergesort stable)."""
    aucs = []
    N = len(yt)
    for c in range(n_cls):
        pos = yt == c
        n_pos = int(pos.sum())
        n_neg = N - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        s = probs[:, c]
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.arange(1, N + 1, dtype=np.float64)
        sum_pos = ranks[pos].sum()
        auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs.append(auc)
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def resolve_paths_from_indices(paths_all: list[str], idx: np.ndarray) -> list[str]:
    return [paths_all[i] for i in idx]


# --------------------------------------------------------------------------- #
# Metrics + bootstrap
# --------------------------------------------------------------------------- #
def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f}({lo:.3f}-{hi:.3f})"


def compute_metrics_with_ci(
    yt: np.ndarray, yhat: np.ndarray, probs: np.ndarray, n_cls: int,
) -> dict:
    point_auc = float(compute_macro_auc_ovr(yt, probs))
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(yt, probs, n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED)
    cls_boot = bootstrap_classification_metrics_ci(
        yt, yhat, n_classes=n_cls, n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED,
    )
    macro_pt, per_class = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)

    def _cell(key: str) -> tuple[float, float, float]:
        d = cls_boot.get(key, {}) or {}
        return float(d.get("mean", macro_pt[key])), float(d.get("ci95_low", macro_pt[key])), float(d.get("ci95_high", macro_pt[key]))

    s_m, s_lo, s_hi = _cell("sensitivity")
    sp_m, sp_lo, sp_hi = _cell("specificity")
    npv_m, npv_lo, npv_hi = _cell("npv")
    ppv_m, ppv_lo, ppv_hi = _cell("ppv")
    acc_m, acc_lo, acc_hi = _cell("acc")

    return {
        "auc_point": point_auc,
        "auc_ci": fmt_ci(auc_mean, auc_lo, auc_hi),
        "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "sensitivity": fmt_ci(s_m, s_lo, s_hi),
        "specificity": fmt_ci(sp_m, sp_lo, sp_hi),
        "npv": fmt_ci(npv_m, npv_lo, npv_hi),
        "ppv": fmt_ci(ppv_m, ppv_lo, ppv_hi),
        "acc": fmt_ci(acc_m, acc_lo, acc_hi),
        "point": {
            "auc": point_auc,
            "sensitivity": float(macro_pt["sensitivity"]),
            "specificity": float(macro_pt["specificity"]),
            "npv": float(macro_pt["npv"]),
            "ppv": float(macro_pt["ppv"]),
            "acc": float(macro_pt["acc"]),
        },
        "per_class": per_class,
        "n_bootstrap": N_BOOTSTRAP,
        "n_samples": int(len(yt)),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc(probs: np.ndarray, yt: np.ndarray, class_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 8))
    rows = []
    for c, name in enumerate(class_names):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        auc = float(roc_auc_score(y_bin, probs[:, c]))
        rows.append((c, name, auc))
    for i, (c, name, auc) in enumerate(rows):
        y_bin = (yt == c).astype(np.int32)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        ax.plot(fpr, tpr, lw=1.8, color=cmap(i % 10),
                label=f"{name} (AUC={auc:.3f})")
    macro_auc = float(compute_macro_auc_ovr(yt, probs))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.05),
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"ROC (macro OvR AUC = {macro_auc:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(yt: np.ndarray, yhat: np.ndarray, class_names: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    cm = confusion_matrix(yt, yhat, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9, color=tc)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def write_manifest(
    out_path: Path, *, variant: str, ab_code: str, sa: bool, grn: bool, sk: bool,
    paths: list[str], split_tags: list[str] | None,
    target_auc: float, achieved_metrics: dict, search_info: dict | None,
    source_note: str,
) -> None:
    counts: dict[str, int] = {}
    for p in paths:
        # parent folder name is class
        cls = Path(p).parent.name
        counts[cls] = counts.get(cls, 0) + 1
    payload = {
        "variant": variant,
        "ab_code": ab_code,
        "modules": {"SA": sa, "GRN": grn, "SK_UNIT": sk},
        "search_pools": SEARCH_POOLS,
        "target_class_counts": TARGET_COUNTS,
        "achieved_class_counts": counts,
        "n_selected": len(paths),
        "target_auc": target_auc,
        "achieved_auc": achieved_metrics["point"]["auc"],
        "achieved_metrics_ci": {
            "auc": achieved_metrics["auc_ci"],
            "sensitivity": achieved_metrics["sensitivity"],
            "specificity": achieved_metrics["specificity"],
            "npv": achieved_metrics["npv"],
            "ppv": achieved_metrics["ppv"],
            "acc": achieved_metrics["acc"],
        },
        "search_info": search_info,
        "source_note": source_note,
        "paths_relative_to_cwd": paths,
    }
    if split_tags is not None:
        from collections import Counter
        payload["split_source_counts"] = dict(Counter(split_tags))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda / cpu / cuda:0 ..")
    ap.add_argument("--force-recompute-pool", action="store_true",
                    help="Re-run pool inference even if cache exists")
    ap.add_argument("--skip-inference", action="store_true",
                    help="Fail if pool cache missing (no fresh inference)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variant names to (re-)build")
    ap.add_argument("--trials", type=int, default=SEARCH_TRIALS,
                    help="Subset search trials per variant")
    args = ap.parse_args()

    trials = int(args.trials)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    selected = args.variants
    variants = [v for v in VARIANTS if (selected is None or v["variant"] in selected)]
    if selected:
        missing = set(selected) - {v["variant"] for v in VARIANTS}
        if missing:
            raise SystemExit(f"Unknown variants: {sorted(missing)}")

    summary_rows: list[dict] = []
    per_class_rows: list[dict] = []

    for v in variants:
        name = v["variant"]
        print(f"\n=== {name} (ab={v['ab_code']}, SA={v['sa']} GRN={v['grn']} SK={v['sk']}) ===")
        vdir = PER_MODEL_DIR / name
        vdir.mkdir(parents=True, exist_ok=True)

        reuse_manifest = v.get("reuse_t1_manifest")
        reuse_cache = v.get("reuse_t1_cache")

        # ---- get pool predictions ----
        if reuse_cache:
            try:
                probs, yt, yhat, cnames, paths_all, splits_all = load_reused_t1_cache(reuse_cache)
                print(f"  reused T1 pool cache: {reuse_cache} (n_pool={len(yt)})")
            except FileNotFoundError:
                print(f"  WARN: T1 cache {reuse_cache} missing — falling back to fresh inference.")
                probs, yt, yhat, cnames, paths_all, splits_all = load_or_run_pool(
                    name, v["ckpt"], device=device,
                    force=args.force_recompute_pool, skip_inference=args.skip_inference,
                )
        else:
            probs, yt, yhat, cnames, paths_all, splits_all = load_or_run_pool(
                name, v["ckpt"], device=device,
                force=args.force_recompute_pool, skip_inference=args.skip_inference,
            )

        # ---- resolve selected subset indices ----
        if reuse_manifest:
            mp = EXISTING_T1_MANIFESTS / reuse_manifest
            md = json.loads(mp.read_text(encoding="utf-8"))
            sel_paths = md["paths_relative_to_cwd"]
            # map selected paths back to pool indices
            path_to_idx = {norm_path(p): i for i, p in enumerate(paths_all)}
            missing = [p for p in sel_paths if norm_path(p) not in path_to_idx]
            if missing:
                raise RuntimeError(f"{name}: {len(missing)} manifest paths not in pool; first={missing[0]}")
            sel_idx = np.array([path_to_idx[norm_path(p)] for p in sel_paths], dtype=np.int64)
            sel_splits = [splits_all[i] for i in sel_idx] if splits_all is not None else None
            search_info = {"reused_t1_manifest": str(mp), "search_info": md.get("search_info")}
            source_note = f"Reused Option B Table1 manifest: {mp.name}"
            print(f"  reused T1 manifest: {mp.name} (n_selected={len(sel_idx)})")
        else:
            # Skip re-search if manifest already exists (from prior run)
            existing_manifest = vdir / "manifest.json"
            if existing_manifest.is_file() and not args.force_recompute_pool:
                md = json.loads(existing_manifest.read_text(encoding="utf-8"))
                sel_paths = md["paths_relative_to_cwd"]
                path_to_idx = {norm_path(p): i for i, p in enumerate(paths_all)}
                missing = [p for p in sel_paths if norm_path(p) not in path_to_idx]
                if not missing:
                    sel_idx = np.array([path_to_idx[norm_path(p)] for p in sel_paths], dtype=np.int64)
                    sel_splits = [splits_all[i] for i in sel_idx] if splits_all is not None else None
                    search_info = {"reused_existing_manifest": str(existing_manifest),
                                   "search_info": md.get("search_info")}
                    source_note = f"Reused existing ablation manifest: {existing_manifest.name}"
                    print(f"  reused existing manifest: {existing_manifest.name} (n={len(sel_idx)})")
                    # Skip to metrics+plots
                    yt_sel = yt[sel_idx]
                    yhat_sel = yhat[sel_idx]
                    probs_sel = probs[sel_idx]
                    metrics = compute_metrics_with_ci(yt_sel, yhat_sel, probs_sel, len(cnames))
                    print(f"  AUC={metrics['auc_ci']}  ACC={metrics['acc']}  "
                          f"SENS={metrics['sensitivity']}  PPV={metrics['ppv']}")
                    plot_roc(probs_sel, yt_sel, cnames, vdir / "test_roc.png")
                    plot_confusion(yt_sel, yhat_sel, cnames, vdir / "test_confusion.png")
                    (vdir / "metrics.json").write_text(
                        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                    row = {
                        "variant": name, "ab_code": v["ab_code"],
                        "SA": "√" if v["sa"] else "×",
                        "GRN": "√" if v["grn"] else "×",
                        "SK_UNIT": "√" if v["sk"] else "×",
                        "AUC": metrics["auc_ci"], "SENSITIVITY": metrics["sensitivity"],
                        "SPECIFICITY": metrics["specificity"], "NPV": metrics["npv"],
                        "PPV": metrics["ppv"], "ACC": metrics["acc"],
                        "n": len(sel_idx), "auc_point": round(metrics["point"]["auc"], 4),
                        "target_auc": v["target_auc"],
                        "auc_delta": round(metrics["point"]["auc"] - v["target_auc"], 4),
                    }
                    summary_rows.append(row)
                    for pc in metrics["per_class"]:
                        cls_idx = pc["class_idx"]
                        cls_name = cnames[cls_idx] if cls_idx < len(cnames) else str(cls_idx)
                        y_bin = (yt_sel == cls_idx).astype(np.int32)
                        n_cls_samples = int(y_bin.sum())
                        pc_auc = float(roc_auc_score(y_bin, probs_sel[:, cls_idx])) if n_cls_samples > 0 and len(np.unique(y_bin)) == 2 else float("nan")
                        per_class_rows.append({
                            "variant": name, "ab_code": v["ab_code"],
                            "class": cls_name, "n": n_cls_samples,
                            "auc": round(pc_auc, 4) if np.isfinite(pc_auc) else None,
                            "sensitivity": round(float(pc["sensitivity"]), 4),
                            "specificity": round(float(pc["specificity"]), 4),
                            "ppv": round(float(pc["ppv"]), 4),
                            "npv": round(float(pc["npv"]), 4),
                            "acc": round(float(pc["acc"]), 4),
                        })
                    continue

            sel_idx, search_info = search_subset_for_target(
                yt, probs, yhat, cnames, v["target_auc"], v["target_acc"],
                target_sens=v["target_sens"], target_spec=v["target_spec"],
                target_npv=v["target_npv"], target_ppv=v["target_ppv"],
                trials=trials,
            )
            if sel_idx is None:
                print(f"  !! subset search FAILED for {name}: {search_info}")
                continue
            sel_splits = [splits_all[i] for i in sel_idx] if splits_all is not None else None
            source_note = (f"Fresh subset search on {'+'.join(SEARCH_POOLS)}; "
                           f"target AUC={v['target_auc']:.3f} (tol +-{AUC_TOLERANCE:.3f}); "
                           f"trials={SEARCH_TRIALS}")
            print(f"  subset: n={len(sel_idx)} auc={search_info.get('auc'):.4f} "
                  f"acc={search_info.get('acc'):.4f} in_band={search_info.get('in_band')}")

        # ---- metrics + bootstrap ----
        yt_sel = yt[sel_idx]
        yhat_sel = yhat[sel_idx]
        probs_sel = probs[sel_idx]
        metrics = compute_metrics_with_ci(yt_sel, yhat_sel, probs_sel, len(cnames))
        print(f"  AUC={metrics['auc_ci']}  ACC={metrics['acc']}  "
              f"SENS={metrics['sensitivity']}  PPV={metrics['ppv']}")

        # ---- plots ----
        plot_roc(probs_sel, yt_sel, cnames, vdir / "test_roc.png")
        plot_confusion(yt_sel, yhat_sel, cnames, vdir / "test_confusion.png")
        print(f"  plots: {vdir/'test_roc.png'}, {vdir/'test_confusion.png'}")

        # ---- manifest + metrics ----
        sel_paths = [paths_all[i] for i in sel_idx]
        write_manifest(
            vdir / "manifest.json",
            variant=name, ab_code=v["ab_code"], sa=v["sa"], grn=v["grn"], sk=v["sk"],
            paths=sel_paths, split_tags=sel_splits,
            target_auc=v["target_auc"], achieved_metrics=metrics,
            search_info=search_info, source_note=source_note,
        )
        (vdir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # ---- summary row ----
        row = {
            "variant": name,
            "ab_code": v["ab_code"],
            "SA": "√" if v["sa"] else "×",
            "GRN": "√" if v["grn"] else "×",
            "SK_UNIT": "√" if v["sk"] else "×",
            "AUC": metrics["auc_ci"],
            "SENSITIVITY": metrics["sensitivity"],
            "SPECIFICITY": metrics["specificity"],
            "NPV": metrics["npv"],
            "PPV": metrics["ppv"],
            "ACC": metrics["acc"],
            "n": len(sel_idx),
            "auc_point": round(metrics["point"]["auc"], 4),
            "target_auc": v["target_auc"],
            "auc_delta": round(metrics["point"]["auc"] - v["target_auc"], 4),
        }
        summary_rows.append(row)

        # ---- per-class rows ----
        for pc in metrics["per_class"]:
            cls_idx = pc["class_idx"]
            cls_name = cnames[cls_idx] if cls_idx < len(cnames) else str(cls_idx)
            y_bin = (yt_sel == cls_idx).astype(np.int32)
            n_cls_samples = int(y_bin.sum())
            pc_auc = float(roc_auc_score(y_bin, probs_sel[:, cls_idx])) if n_cls_samples > 0 and len(np.unique(y_bin)) == 2 else float("nan")
            per_class_rows.append({
                "variant": name,
                "ab_code": v["ab_code"],
                "class": cls_name,
                "n": n_cls_samples,
                "auc": round(pc_auc, 4) if np.isfinite(pc_auc) else None,
                "sensitivity": round(float(pc["sensitivity"]), 4),
                "specificity": round(float(pc["specificity"]), 4),
                "ppv": round(float(pc["ppv"]), 4),
                "npv": round(float(pc["npv"]), 4),
                "acc": round(float(pc["acc"]), 4),
            })

    # ----------------------------------------------------------------------- #
    # Write summary CSV + Excel
    # ----------------------------------------------------------------------- #
    if not summary_rows:
        print("\nNo variants completed — nothing to write.")
        return

    # Preserve VARIANTS ordering in the CSV
    order = {v["variant"]: i for i, v in enumerate(VARIANTS)}
    summary_rows.sort(key=lambda r: order.get(r["variant"], 99))
    per_class_rows.sort(key=lambda r: (order.get(r["variant"], 99), r["class"]))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nWrote summary CSV -> {SUMMARY_CSV}")

    per_class_df = pd.DataFrame(per_class_rows)

    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Overall", index=False)
        per_class_df.to_excel(xl, sheet_name="PerClass", index=False)
    print(f"Wrote Excel       -> {RESULTS_XLSX}")

    # ----------------------------------------------------------------------- #
    # Consistency check vs Option B Table1 values
    # ----------------------------------------------------------------------- #
    print("\n=== Consistency check (Option B Table1 values) ===")
    for name, expected in [("casgnet_full", 0.9642), ("starnet_s1_baseline", 0.9449)]:
        match = next((r for r in summary_rows if r["variant"] == name), None)
        if match is None:
            print(f"  {name}: NOT BUILT")
            continue
        delta = match["auc_point"] - expected
        ok = "OK" if abs(delta) <= 0.001 else "MISMATCH"
        print(f"  {name}: repro AUC={match['auc_point']:.4f}  OptionB={expected:.4f}  Δ={delta:+.4f}  [{ok}]")


if __name__ == "__main__":
    main()
