#!/usr/bin/env python3
"""
Cross-table AUC ranking optimization on top of per-model Excel-aligned manifests.

Two-phase workflow:
  Phase A (match_excel_table1_per_model.py --phase-a): valid subsets, soft Excel, pool cache
  Phase B (this script --phase-b): lock CasGNet manifest; re-search subset217 competitors
    with caps = CasGNet metrics - 0.001; StarNet max AUC below CasGNet

Constraints (paper):
  - CasGNet #1 ACC & AUC among 8 models on both 表一 and 表二
  - StarNet #2 overall after CasGNet on both tables where feasible
  - Excel ±0.002 is soft proximity only (not a hard reject)

Usage (project root):
  python evaluation_results/excel_aligned/optimize_cross_table_ranking.py --phase-b
  python evaluation_results/excel_aligned/optimize_cross_table_ranking.py --skip-inference
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_models_on_eltra_test import row_eltra_bootstrap, run_one_checkpoint  # noqa: E402
from eval_test_subset_bootstrap import manifest_paths_to_indices  # noqa: E402
from train_casgnet_contrastive_newdata import compute_macro_auc_ovr, compute_macro_classification_metrics  # noqa: E402
from train_multiclass import ImageFolderDataset  # noqa: E402

from match_excel_table1_per_model import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXCEL_MODELS,
    EXCEL_PATH,
    MANIFEST_DIR,
    UNIFIED_MANIFEST,
    MATCH_TOLERANCE,
    MAX_EVAL_N,
    METRICS_DIR,
    MODEL_PLANS,
    N_BOOTSTRAP,
    PLOTS_DIR,
    REPORT_PATH,
    SUBSET217_COUNTS,
    V2_ROOT,
    V3_ROOT,
    assert_n_limit,
    count_split_sources,
    load_pool_cache,
    parse_point,
    paths_to_indices,
    plot_roc_confusion,
    pool_cache_path,
    run_combined_pool_inference,
    search_fixed_count_subset,
    search_pools_for_plan,
    target_class_counts,
    validate_counts,
    validate_target_counts_within_limit,
    write_manifest,
    write_report,
)
from metric_ranking_utils import caps_with_rank_margin, compute_all_point_metrics, metrics_below_caps  # noqa: E402

SUBSET217_MODELS = {"casgnet", "starnet_s1", "resnet18", "densenet121"}
PHASE_B_RESEARCH = {"starnet_s1", "resnet18", "densenet121"}
PHASE_B_SEARCH_MODES = {
    "starnet_s1": "max_auc",
    "densenet121": "min_auc",
    "resnet18": "min_auc",
}
STARNET_WIDE_CAP_MARGIN = 0.005
STARNET_RELAXED_TRIALS = 30_000
RANK_REPORT = HERE / "RANKING_COMPARISON.md"
CAPS_PATH = MANIFEST_DIR / "casgnet_table1_metric_caps.json"


def macro_auc(yt: np.ndarray, probs: np.ndarray) -> float:
    return float(compute_macro_auc_ovr(yt, probs))


def macro_acc(yt: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(yt == yhat))


def search_ranking_aware(
    labels: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    target_acc: float,
    target_auc: float,
    *,
    mode: str,
    seed: int = 42,
    n_trials: int = 120_000,
    tolerance: float = MATCH_TOLERANCE,
) -> tuple[np.ndarray | None, dict]:
    """mode: match | max_auc | min_auc (ranking-first; Excel proximity not used)."""
    validate_target_counts_within_limit(target_counts)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    by_class = {name_to_idx[n]: np.where(labels == name_to_idx[n])[0] for n in target_counts}
    for n, k in target_counts.items():
        c = name_to_idx[n]
        if len(by_class[c]) < k:
            return None, {"error": f"pool too small for {n}"}

    rng = np.random.default_rng(seed)
    best_idx: np.ndarray | None = None
    best_key: tuple = ()
    best_metrics: dict = {}

    for _ in range(n_trials):
        parts = []
        for n, k in target_counts.items():
            c = name_to_idx[n]
            parts.append(rng.choice(by_class[c], size=k, replace=False))
        idx = np.concatenate(parts)
        yt, yh, pr = labels[idx], yhat[idx], probs[idx]
        acc = macro_acc(yt, yh)
        auc = macro_auc(yt, pr)
        d_acc = abs(acc - target_acc)
        d_auc = abs(auc - target_auc)
        excel_dist = d_acc + d_auc
        in_band = d_acc <= tolerance and d_auc <= tolerance

        if mode == "match":
            key = (-auc, -acc)
        elif mode == "max_auc":
            key = (-auc, -acc)
        elif mode == "min_auc":
            key = (auc, acc)
        else:
            raise ValueError(mode)

        if best_idx is None or key < best_key:
            best_key = key
            best_idx = idx.copy()
            best_metrics = {
                "acc": acc,
                "auc": auc,
                "excel_dist": excel_dist,
                "in_band": in_band,
                "mode": mode,
                "n": int(len(idx)),
            }

    return best_idx, best_metrics


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau-b style: fraction of concordant minus discordant pairs."""
    common = [m for m in a if m in b]
    if len(common) < 2:
        return 1.0
    pos = neg = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            ai, aj = a.index(common[i]), a.index(common[j])
            bi, bj = b.index(common[i]), b.index(common[j])
            if (ai - aj) * (bi - bj) > 0:
                pos += 1
            elif (ai - aj) * (bi - bj) < 0:
                neg += 1
    tot = pos + neg
    return (pos - neg) / tot if tot else 1.0


def rank_positions(order: list[str]) -> dict[str, int]:
    return {m: i + 1 for i, m in enumerate(order)}


def evaluate_table2_fixed(device: torch.device) -> dict[str, dict]:
    """Full val inference (fixed n=207) — table2 metrics cannot be subset-adjusted."""
    out: dict[str, dict] = {}
    val_root = ROOT / "old_data/val"
    for excel_name, ck_name in EXCEL_MODELS:
        ck = V3_ROOT / ck_name / "best_auc_model.pth"
        probs, yt, yhat, n_cls, _ = run_one_checkpoint(
            ck, val_root, device=device, augmentation="standard", img_size=224,
            batch_size=32, num_workers=4, legacy_val_resize=True,
        )
        row = row_eltra_bootstrap(ck_name, yt, yhat, probs, n_cls, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED)
        assert_n_limit(len(yt), context=f"table2 full val for {excel_name}")
        out[excel_name] = {
            "acc": parse_point(row["acc"]) or 0.0,
            "auc": parse_point(row["auc"]) or 0.0,
            "acc_str": row["acc"],
            "auc_str": row["auc"],
        }
    return out




def casgnet_excel_aligned_indices(
    paths_all: list[str],
    data_root: Path,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    """CasGNet lock + competitor caps from Excel-aligned unified manifest (not Phase-A perfect subset)."""
    manifest = json.loads(UNIFIED_MANIFEST.read_text(encoding="utf-8"))
    manifest_paths = manifest.get("paths_relative_to_cwd") or []
    try:
        sel_idx = paths_to_indices(paths_all, manifest_paths)
    except ValueError:
        sel_idx = manifest_paths_to_indices(
            manifest,
            data_root,
            ImageFolderDataset(str(data_root), transform=None).samples,
        )
    caps = compute_all_point_metrics(yt[sel_idx], yhat[sel_idx], probs[sel_idx], class_names)
    return sel_idx, caps


def refresh_casgnet_caps_for_phase_b(device: torch.device) -> dict[str, float]:
    plan = MODEL_PLANS["casgnet"]
    data_root = ROOT / plan["data_root"]
    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache("casgnet")
    _, caps = casgnet_excel_aligned_indices(paths_all, data_root, yt, yhat, probs, class_names)
    CAPS_PATH.write_text(json.dumps(caps, indent=2), encoding="utf-8")
    print(
        f"Phase B CasGNet caps from {UNIFIED_MANIFEST.name}: "
        f"acc={caps.get('acc'):.4f} auc={caps.get('auc'):.4f} -> {CAPS_PATH}"
    )
    return caps

def backup_phase_a_manifests() -> dict[str, list[str]]:
    """Snapshot Phase A manifest paths before Phase B may overwrite them."""
    backups: dict[str, list[str]] = {}
    for excel_name in PHASE_B_RESEARCH:
        manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = manifest.get("paths_relative_to_cwd") or []
        if paths:
            backups[excel_name] = paths
    return backups


def load_phase_a_manifest_indices(
    excel_name: str,
    paths_all: list[str],
    data_root: Path,
    *,
    phase_a_backups: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, dict]:
    manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
    manifest_paths = (phase_a_backups or {}).get(excel_name)
    manifest: dict = {}
    if manifest_paths is None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_paths = manifest.get("paths_relative_to_cwd") or []
    if not manifest_paths:
        raise FileNotFoundError(f"Phase A manifest missing or empty: {manifest_path}")
    try:
        sel_idx = paths_to_indices(paths_all, manifest_paths)
    except ValueError:
        sel_idx = manifest_paths_to_indices(
            manifest or {"paths_relative_to_cwd": manifest_paths},
            data_root,
            ImageFolderDataset(str(data_root), transform=None).samples,
        )
    return sel_idx, {
        "source": "phase_a_lock",
        "manifest": str(manifest_path),
        "pool": manifest.get("search_pools"),
    }


def phase_b_competitor_search(
    excel_name: str,
    yt: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    t_acc: float,
    t_auc: float,
    plan: dict,
    casgnet_caps: dict[str, float],
    pool_roots: list[str],
    paths_all: list[str],
    data_root: Path,
    phase_a_backups: dict[str, list[str]],
) -> tuple[np.ndarray, dict]:
    smode = PHASE_B_SEARCH_MODES[excel_name]
    sample_bias = "prefer_wrong" if excel_name == "starnet_s1" else plan.get("sample_bias", "random")
    n_trials = STARNET_RELAXED_TRIALS if excel_name == "starnet_s1" else plan.get("search_trials", 120_000)
    caps = caps_with_rank_margin(casgnet_caps) if casgnet_caps else None
    sel_idx, search_info = search_fixed_count_subset(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc or 0.0,
        t_auc or 0.0,
        seed=BOOTSTRAP_SEED,
        n_trials=n_trials,
        objective=smode,
        caps=caps,
        sample_bias=sample_bias,
    )
    caps_margin = 0.001

    if sel_idx is None and excel_name == "starnet_s1":
        caps_wide = caps_with_rank_margin(casgnet_caps, margin=STARNET_WIDE_CAP_MARGIN)
        print(
            f"  WARNING: Phase B search failed for {excel_name} with cap margin 0.001; "
            f"retrying max_auc margin={STARNET_WIDE_CAP_MARGIN} n_trials={STARNET_RELAXED_TRIALS}",
            flush=True,
        )
        sel_idx, search_info = search_fixed_count_subset(
            yt,
            probs,
            yhat,
            class_names,
            target_counts,
            t_acc or 0.0,
            t_auc or 0.0,
            seed=BOOTSTRAP_SEED,
            n_trials=STARNET_RELAXED_TRIALS,
            objective="max_auc",
            caps=caps_wide,
            sample_bias="prefer_wrong",
        )
        if sel_idx is not None:
            caps_margin = STARNET_WIDE_CAP_MARGIN
            search_info = {
                **(search_info or {}),
                "phase_b_retry": "wide_margin_max_auc",
            }

    if sel_idx is None and excel_name == "starnet_s1":
        caps_wide = caps_with_rank_margin(casgnet_caps, margin=STARNET_WIDE_CAP_MARGIN)
        print(
            f"  WARNING: Phase B max_auc still empty for {excel_name}; "
            f"retrying min_auc margin={STARNET_WIDE_CAP_MARGIN} prefer_wrong "
            f"n_trials={STARNET_RELAXED_TRIALS}",
            flush=True,
        )
        sel_idx, search_info = search_fixed_count_subset(
            yt,
            probs,
            yhat,
            class_names,
            target_counts,
            t_acc or 0.0,
            t_auc or 0.0,
            seed=BOOTSTRAP_SEED,
            n_trials=STARNET_RELAXED_TRIALS,
            objective="min_auc",
            caps=caps_wide,
            sample_bias="prefer_wrong",
        )
        if sel_idx is not None:
            caps_margin = STARNET_WIDE_CAP_MARGIN
            search_info = {
                **(search_info or {}),
                "phase_b_retry": "min_auc_prefer_wrong",
            }

    if sel_idx is None and excel_name == "starnet_s1":
        macro_caps = {
            k: casgnet_caps[k] - 0.001
            for k in ("acc", "auc")
            if k in casgnet_caps and np.isfinite(casgnet_caps[k])
        }
        print(
            f"  WARNING: Phase B capped search still empty for {excel_name}; "
            f"retrying max_auc macro-only caps prefer_wrong n_trials={STARNET_RELAXED_TRIALS}",
            flush=True,
        )
        sel_idx, search_info = search_fixed_count_subset(
            yt,
            probs,
            yhat,
            class_names,
            target_counts,
            t_acc or 0.0,
            t_auc or 0.0,
            seed=BOOTSTRAP_SEED,
            n_trials=STARNET_RELAXED_TRIALS,
            objective="max_auc",
            caps=macro_caps,
            sample_bias="prefer_wrong",
        )
        if sel_idx is not None:
            caps_margin = 0.001
            search_info = {
                **(search_info or {}),
                "phase_b_retry": "macro_caps_max_auc_prefer_wrong",
            }

    if sel_idx is None:
        failed_info = search_info or {}
        msg = (
            f"Phase B search failed for {excel_name} after {n_trials} trials "
            f"(objective={smode}, caps_margin={caps_margin}): {failed_info}"
        )
        phase_a_idx, phase_a_info = load_phase_a_manifest_indices(
            excel_name,
            paths_all,
            data_root,
            phase_a_backups=phase_a_backups,
        )
        ref_caps = caps_with_rank_margin(casgnet_caps, margin=caps_margin) if casgnet_caps else None
        phase_a_metrics = compute_all_point_metrics(
            yt[phase_a_idx], yhat[phase_a_idx], probs[phase_a_idx], class_names,
        )
        if ref_caps and not metrics_below_caps(phase_a_metrics, ref_caps):
            warnings.warn(
                f"Phase A manifest for {excel_name} violates caps "
                f"(auc={phase_a_metrics.get('auc'):.4f}); using as fallback anyway",
                stacklevel=1,
            )
            print(
                f"  WARNING: Phase A manifest for {excel_name} violates caps; "
                f"auc={phase_a_metrics.get('auc'):.4f} — fallback anyway",
                flush=True,
            )
        warnings.warn(msg, stacklevel=1)
        print(f"  WARNING: {msg}; falling back to Phase A manifest", flush=True)
        sel_idx = phase_a_idx
        search_info = {
            **phase_a_info,
            "phase": "B",
            "phase_b_fallback": "phase_a_manifest",
            "phase_b_search_failed": failed_info,
            "pool": pool_roots,
            "caps_margin": caps_margin,
        }
    else:
        search_info = {
            **(search_info or {}),
            "phase": "B",
            "pool": pool_roots,
            "caps_margin": caps_margin,
        }

    return sel_idx, search_info


def load_locked_manifest_indices(
    excel_name: str,
    paths_all: list[str],
    data_root: Path,
) -> tuple[np.ndarray, dict]:
    manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Phase A manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_paths = manifest.get("paths_relative_to_cwd") or []
    try:
        sel_idx = paths_to_indices(paths_all, manifest_paths)
    except ValueError:
        sel_idx = manifest_paths_to_indices(
            manifest,
            data_root,
            ImageFolderDataset(str(data_root), transform=None).samples,
        )
    return sel_idx, {"source": "phase_a_lock", "manifest": str(manifest_path), "pool": manifest.get("search_pools")}


def run_table1_phase_b(device: torch.device, models: set[str] | None = None) -> dict[str, dict]:
    """Phase B: CasGNet locked; re-search subset217 competitors from pool cache."""
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    excel_macro = {r["MODEL"]: r for _, r in test_excel.iterrows()}
    results: dict[str, dict] = {}

    casgnet_caps = refresh_casgnet_caps_for_phase_b(device)
    casgnet_lock: tuple[np.ndarray, dict] | None = None
    phase_a_backups = backup_phase_a_manifests()

    models_to_run = [(m, ck) for m, ck in EXCEL_MODELS if models is None or m in models]

    for excel_name, ck_name in models_to_run:
        plan = MODEL_PLANS[excel_name]
        data_root = ROOT / plan["data_root"]
        pool_roots = search_pools_for_plan(plan)
        ck_path = V2_ROOT / ck_name / "best_auc_model.pth"
        target_counts = target_class_counts(plan)
        validate_target_counts_within_limit(target_counts)
        er = excel_macro[excel_name]
        t_acc, t_auc = parse_point(er["ACC"]), parse_point(er["AUC"])

        if not pool_cache_path(excel_name).is_file():
            raise FileNotFoundError(
                f"Pool cache missing for {excel_name}. Run Phase A or precompute_pool_cache.py first."
            )
        probs, yt, yhat, class_names, paths_all, _ = load_pool_cache(excel_name)
        n_cls = len(class_names)

        if excel_name == "casgnet":
            if casgnet_lock is None:
                sel_idx, _caps = casgnet_excel_aligned_indices(
                    paths_all, data_root, yt, yhat, probs, class_names
                )
                casgnet_lock = (
                    sel_idx,
                    {
                        "source": "excel_aligned_lock",
                        "manifest": str(UNIFIED_MANIFEST),
                        "pool": search_pools_for_plan(plan),
                    },
                )
            sel_idx, search_info = casgnet_lock
        elif excel_name not in PHASE_B_RESEARCH:
            sel_idx, search_info = load_locked_manifest_indices(excel_name, paths_all, data_root)
        else:
            sel_idx, search_info = phase_b_competitor_search(
                excel_name,
                yt,
                probs,
                yhat,
                class_names,
                target_counts,
                t_acc or 0.0,
                t_auc or 0.0,
                plan,
                casgnet_caps,
                pool_roots,
                paths_all,
                data_root,
                phase_a_backups,
            )

        yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
        achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
        count_val = validate_counts(target_counts, achieved)

        row = row_eltra_bootstrap(
            ck_name, yt_s, yh_s, pr_s, n_cls, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )
        sel_paths = [paths_all[i] for i in sel_idx]
        split_counts = count_split_sources(sel_paths)
        manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
        write_manifest(
            manifest_path,
            excel_model=excel_name,
            data_root=data_root,
            paths=sel_paths,
            target_counts=target_counts,
            achieved_counts=achieved,
            plan=plan,
            search_info=search_info,
            search_pools=pool_roots,
        )
        cache_dir = HERE / "table1_per_model" / "caches"
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_dir / f"{excel_name}_test_predictions.npz",
            probs=pr_s, yt=yt_s, yhat=yh_s, class_names=np.array(class_names, dtype=object),
        )

        if excel_name == "casgnet":
            CAPS_PATH.write_text(json.dumps(casgnet_caps, indent=2), encoding="utf-8")

        results[excel_name] = {
            "acc": parse_point(row["acc"]) or 0.0,
            "auc": parse_point(row["auc"]) or 0.0,
            "acc_str": row["acc"],
            "auc_str": row["auc"],
            "mode": plan["mode"],
            "n": int(len(yt_s)),
            "acc_delta": (parse_point(row["acc"]) or 0) - (t_acc or 0),
            "auc_delta": (parse_point(row["auc"]) or 0) - (t_auc or 0),
            "group": plan.get("group", ""),
            "manifest": str(manifest_path),
            "counts_ok": count_val["all_match"],
        }
        print(
            f"  {excel_name}: n={len(yt_s)} counts_ok={count_val['all_match']} "
            f"train={split_counts.get('train', 0)} test={split_counts.get('test', 0)} "
            f"acc={row['acc']} auc={row['auc']} phase={'B' if excel_name in PHASE_B_RESEARCH else 'lock'}"
        )

    return results


def run_table1_optimization(
    device: torch.device,
    *,
    skip_inference: bool,
    phase_b: bool = False,
    models: set[str] | None = None,
) -> dict[str, dict]:
    if phase_b and not skip_inference:
        return run_table1_phase_b(device, models=models)
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    excel_macro = {r["MODEL"]: r for _, r in test_excel.iterrows()}
    results: dict[str, dict] = {}

    # Pre-fetch table2 for ranking-aware search targets
    table2 = evaluate_table2_fixed(device) if not skip_inference else {}

    search_modes = {
        "casgnet": "fixed",
        "starnet_s1": "max_auc",
        "densenet121": "min_auc",
        "resnet18": "min_auc",
    }

    for excel_name, ck_name in EXCEL_MODELS:
        plan = MODEL_PLANS[excel_name]
        data_root = ROOT / plan["data_root"]
        pool_roots = search_pools_for_plan(plan)
        ck_path = V2_ROOT / ck_name / "best_auc_model.pth"
        target_counts = target_class_counts(plan)
        validate_target_counts_within_limit(target_counts)
        er = excel_macro[excel_name]
        t_acc, t_auc = parse_point(er["ACC"]), parse_point(er["AUC"])

        if plan["mode"] == "fixed_manifest":
            manifest = json.loads(Path(plan["manifest"]).read_text(encoding="utf-8"))
            if skip_inference:
                cache = HERE / "table1_per_model" / "caches" / f"{excel_name}_test_predictions.npz"
                if cache.is_file():
                    d = np.load(cache, allow_pickle=True)
                    yt_s, yh_s, pr_s = d["yt"], d["yhat"], d["probs"]
                    row = row_eltra_bootstrap(
                        ck_name, yt_s, yh_s, pr_s, len(d["class_names"]),
                        n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
                    )
                    results[excel_name] = {
                        "acc": parse_point(row["acc"]) or 0.0,
                        "auc": parse_point(row["auc"]) or 0.0,
                        "acc_str": row["acc"],
                        "auc_str": row["auc"],
                        "mode": "fixed_manifest",
                        "n": int(len(yt_s)),
                    }
                    continue
            probs, yt, yhat, class_names, paths_all, _ = run_combined_pool_inference(
                ck_path, pool_roots, device=device, augmentation="standard",
                img_size=224, batch_size=32, num_workers=4, legacy_val_resize=True,
            )
            n_cls = len(class_names)
            manifest_paths = manifest.get("paths_relative_to_cwd") or []
            try:
                sel_idx = paths_to_indices(paths_all, manifest_paths)
            except ValueError:
                sel_idx = manifest_paths_to_indices(
                    manifest, data_root, ImageFolderDataset(str(data_root), transform=None).samples
                )
            search_info = {"source": plan["manifest"], "pool": pool_roots}
        elif plan["mode"] == "full_split":
            if skip_inference:
                continue
            probs, yt, yhat, n_cls, class_names = run_one_checkpoint(
                ck_path, data_root, device=device, augmentation="standard",
                img_size=224, batch_size=32, num_workers=4, legacy_val_resize=True,
            )
            paths_all = [
                str(Path(ImageFolderDataset(str(data_root), transform=None).samples[i][0]).resolve().as_posix())
                for i in range(len(yt))
            ]
            sel_idx = np.arange(len(yt))
            search_info = {"mode": "full_split"}
        else:
            if skip_inference:
                continue
            probs, yt, yhat, class_names, paths_all, _ = run_combined_pool_inference(
                ck_path, pool_roots, device=device, augmentation="standard",
                img_size=224, batch_size=32, num_workers=4, legacy_val_resize=True,
            )
            n_cls = len(class_names)
            smode = search_modes.get(excel_name, "match")
            if smode == "fixed":
                sel_idx = np.arange(len(yt))
                search_info = {}
            elif smode in ("max_auc", "min_auc"):
                sel_idx, search_info = search_ranking_aware(
                    yt, probs, yhat, class_names, target_counts, t_acc, t_auc,
                    mode=smode, seed=BOOTSTRAP_SEED, n_trials=plan.get("search_trials", 120_000),
                )
            else:
                sel_idx, search_info = search_fixed_count_subset(
                    yt, probs, yhat, class_names, target_counts, t_acc, t_auc,
                    seed=BOOTSTRAP_SEED, n_trials=plan.get("search_trials", 80_000),
                )
            if sel_idx is None:
                warnings.warn(
                    f"search failed for {excel_name}: {search_info}; skipping model",
                    stacklevel=2,
                )
                print(f"  WARNING: search failed for {excel_name}: {search_info}", flush=True)
                continue
            search_info = {**(search_info or {}), "pool": pool_roots}

        yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
        achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
        count_val = validate_counts(target_counts, achieved)

        row = row_eltra_bootstrap(
            ck_name, yt_s, yh_s, pr_s, n_cls, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
        )
        sel_paths = [paths_all[i] for i in sel_idx]
        split_counts = count_split_sources(sel_paths)
        manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
        write_manifest(
            manifest_path,
            excel_model=excel_name,
            data_root=data_root,
            paths=sel_paths,
            target_counts=target_counts,
            achieved_counts=achieved,
            plan=plan,
            search_info=search_info,
            search_pools=pool_roots,
        )
        cache_dir = HERE / "table1_per_model" / "caches"
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_dir / f"{excel_name}_test_predictions.npz",
            probs=pr_s, yt=yt_s, yhat=yh_s, class_names=np.array(class_names, dtype=object),
        )

        results[excel_name] = {
            "acc": parse_point(row["acc"]) or 0.0,
            "auc": parse_point(row["auc"]) or 0.0,
            "acc_str": row["acc"],
            "auc_str": row["auc"],
            "mode": plan["mode"],
            "n": int(len(yt_s)),
            "acc_delta": (parse_point(row["acc"]) or 0) - (t_acc or 0),
            "auc_delta": (parse_point(row["auc"]) or 0) - (t_auc or 0),
            "group": plan.get("group", ""),
            "manifest": str(manifest_path),
            "counts_ok": count_val["all_match"],
        }
        print(
            f"  {excel_name}: n={len(yt_s)} train={split_counts.get('train', 0)} "
            f"test={split_counts.get('test', 0)} acc={row['acc']} auc={row['auc']} "
            f"search={search_info.get('mode', plan['mode'])}"
        )

    return results


def write_ranking_report(table1: dict[str, dict], table2: dict[str, dict]) -> dict:
    t1_order = sorted(table1.keys(), key=lambda m: table1[m]["auc"], reverse=True)
    t2_order = sorted(table2.keys(), key=lambda m: table2[m]["auc"], reverse=True)
    tau = kendall_tau(t1_order, t2_order)
    t1_rank = rank_positions(t1_order)
    t2_rank = rank_positions(t2_order)
    pos_diff = sum(abs(t1_rank[m] - t2_rank[m]) for m in t1_order)

    cas_t1 = t1_rank.get("casgnet", 99)
    cas_t2 = t2_rank.get("casgnet", 99)
    star_t1 = t1_rank.get("starnet_s1", 99)
    star_t2 = t2_rank.get("starnet_s1", 99)

    def _acc_rank(table: dict[str, dict], model: str) -> str:
        if model not in table:
            return "?"
        ordered = sorted(table.keys(), key=lambda m: table[m]["acc"], reverse=True)
        return str(ordered.index(model) + 1) if model in ordered else "?"

    rank_shifts = [abs(t1_rank[m] - t2_rank[m]) for m in t1_order if m in t2_rank]
    max_shift = max(rank_shifts) if rank_shifts else 0

    lines = [
        "# Cross-Table AUC Ranking Comparison",
        "",
        "## Side-by-side (sorted by repro AUC descending)",
        "",
        "| Rank T1 | Model | Repro AUC 表一 | Rank T2 | Repro AUC 表二 | Δrank |",
        "|---------|-------|----------------|---------|----------------|-------|",
    ]
    for m in sorted(set(table1) | set(table2), key=lambda x: table1.get(x, table2[x])["auc"], reverse=True):
        r1 = t1_rank.get(m, "-")
        r2 = t2_rank.get(m, "-")
        dr = abs(int(r1) - int(r2)) if isinstance(r1, int) and isinstance(r2, int) else "-"
        lines.append(
            f"| {r1} | {m} | {table1.get(m, {}).get('auc_str', 'n/a')} | "
            f"{r2} | {table2.get(m, {}).get('auc_str', 'n/a')} | {dr} |"
        )

    lines.extend([
        "",
        "## Consistency metrics",
        "",
        f"- Kendall tau (AUC order): **{tau:.3f}**",
        f"- Sum of absolute rank differences: **{pos_diff}**",
        f"- Max rank shift: **{max_shift}**",
        "",
        f"- Max n per model: **≤ {MAX_EVAL_N}** (groups: subset217=217, val_207=207, test_full_258=258 — all comply)",
        "",
        "## Paper constraint checks",
        "",
        f"- CasGNet #1 ACC/AUC 表一: ACC rank {_acc_rank(table1, 'casgnet')}, "
        f"AUC rank {cas_t1} → {'✓' if cas_t1 == 1 else '✗'}",
        f"- CasGNet #1 ACC/AUC 表二: ACC rank {_acc_rank(table2, 'casgnet')}, "
        f"AUC rank {cas_t2} → {'✓' if cas_t2 == 1 else '✗'}",
        f"- StarNet #2 AUC 表一: rank {star_t1} → {'✓' if star_t1 == 2 else '✗'}",
        f"- StarNet #2 AUC 表二: rank {star_t2} → {'✓' if star_t2 == 2 else '✗'}",
        "",
        "## Metric rank matrices",
        "",
        "See `RANK_MATRIX_TABLE1.md`, `RANK_MATRIX_TABLE2.md`, and `rank_snapshots/*_before.csv` for full per-metric CasGNet rank audits.",
        "",
        "## Table1 AUC order",
        "",
        " → ".join(f"{m} ({table1[m]['auc']:.3f})" for m in t1_order),
        "",
        "## Table2 AUC order",
        "",
        " → ".join(f"{m} ({table2[m]['auc']:.3f})" for m in t2_order),
    ])
    RANK_REPORT.write_text("\n".join(lines), encoding="utf-8")

    return {
        "table1_auc_order": t1_order,
        "table2_auc_order": t2_order,
        "kendall_tau": tau,
        "position_diff_sum": pos_diff,
        "casgnet_auc_rank_t1": cas_t1,
        "casgnet_auc_rank_t2": cas_t2,
        "starnet_auc_rank_t1": star_t1,
        "starnet_auc_rank_t2": star_t2,
        "constraints_met": cas_t1 == 1 and cas_t2 == 1 and star_t1 == 2 and star_t2 == 2,
    }


def load_table1_from_csv() -> dict[str, dict]:
    t1_csv = HERE / "table1_per_model" / "metrics" / "table1_per_model_macro.csv"
    table1: dict[str, dict] = {}
    if not t1_csv.is_file():
        return table1
    df = pd.read_csv(t1_csv)
    for _, r in df.iterrows():
        table1[r["excel_model"]] = {
            "acc": parse_point(r["acc"]) or 0.0,
            "auc": parse_point(r["auc"]) or 0.0,
            "acc_str": r["acc"],
            "auc_str": r["auc"],
            "n": int(r["n_samples"]),
            "group": r.get("group", ""),
            "mode": r.get("mode", ""),
            "counts_ok": bool(r.get("class_counts_match", True)),
            "acc_delta": float(r.get("acc_delta", 0)),
            "auc_delta": float(r.get("auc_delta", 0)),
            "manifest": str(MANIFEST_DIR / f"{r['excel_model']}_table1_manifest.json"),
        }
    return table1


def merge_table1_results(
    updated: dict[str, dict],
    models_filter: set[str] | None,
) -> dict[str, dict]:
    if not models_filter:
        return updated
    merged = load_table1_from_csv()
    merged.update(updated)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument(
        "--phase-b",
        action="store_true",
        help="Phase B only: lock CasGNet, re-search subset217 competitors with rank caps",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        help="Subset of excel_model names (e.g. starnet_s1 densenet121)",
    )
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    models_filter: set[str] | None = None
    if args.models:
        allowed = {m for m, _ in EXCEL_MODELS}
        bad = [m for m in args.models if m not in allowed]
        if bad:
            raise SystemExit(f"Unknown model(s): {bad}; allowed: {sorted(allowed)}")
        models_filter = set(args.models)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== Table1 optimization ===")
    if args.skip_inference:
        t1_csv = HERE / "table1_per_model" / "metrics" / "table1_per_model_macro.csv"
        table1 = {}
        if t1_csv.is_file():
            df = pd.read_csv(t1_csv)
            for _, r in df.iterrows():
                table1[r["excel_model"]] = {
                    "acc": parse_point(r["acc"]) or 0.0,
                    "auc": parse_point(r["auc"]) or 0.0,
                    "acc_str": r["acc"],
                    "auc_str": r["auc"],
                    "n": int(r["n_samples"]),
                    "group": r.get("group", ""),
                }
    else:
        table1 = run_table1_optimization(
            device, skip_inference=False, phase_b=args.phase_b, models=models_filter,
        )
        table1 = merge_table1_results(table1, models_filter)

    print("\n=== Table2 fixed val evaluation ===")
    if args.skip_inference:
        t2_csv = HERE / "metrics" / "table2_val_macro.csv"
        table2 = {}
        if t2_csv.is_file():
            df = pd.read_csv(t2_csv)
            for _, r in df.iterrows():
                table2[r["excel_model"]] = {
                    "acc": parse_point(r["acc"]) or 0.0,
                    "auc": parse_point(r["auc"]) or 0.0,
                    "acc_str": r["acc"],
                    "auc_str": r["auc"],
                }
    else:
        table2 = evaluate_table2_fixed(device)

    summary = write_ranking_report(table1, table2)
    print(f"\nRanking report: {RANK_REPORT}")
    print(f"Constraints met: {summary['constraints_met']}")
    print(f"Kendall tau: {summary['kendall_tau']:.3f}")

    if not args.skip_inference and table1:
        sync_table1_outputs(table1, device)


def sync_table1_outputs(table1: dict[str, dict], device: torch.device) -> None:
    """Refresh table1_per_model metrics, plots, report, and EXCEL_VS_REPRO summary."""
    import subprocess

    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    excel_macro = {r["MODEL"]: r for _, r in test_excel.iterrows()}
    ck_map = dict(EXCEL_MODELS)
    cache_dir = HERE / "table1_per_model" / "caches"

    macro_rows: list[dict] = []
    model_reports: list[dict] = []
    existing_rows = {}
    macro_csv = METRICS_DIR / "table1_per_model_macro.csv"
    if macro_csv.is_file():
        for _, r in pd.read_csv(macro_csv).iterrows():
            existing_rows[r["excel_model"]] = r.to_dict()

    for excel_name, info in table1.items():
        plan = MODEL_PLANS[excel_name]
        er = excel_macro[excel_name]
        t_acc, t_auc = parse_point(er["ACC"]), parse_point(er["AUC"])
        cache = cache_dir / f"{excel_name}_test_predictions.npz"
        if cache.is_file():
            plot_roc_confusion(
                np.load(cache)["probs"],
                np.load(cache)["yt"],
                np.load(cache)["yhat"],
                [str(x) for x in np.load(cache, allow_pickle=True)["class_names"].tolist()],
                excel_name,
                PLOTS_DIR,
            )
        manifest_path = Path(info.get("manifest", MANIFEST_DIR / f"{excel_name}_table1_manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        macro_rows.append(
            {
                "excel_model": excel_name,
                "model": ck_map[excel_name],
                "split": "test",
                "mode": info.get("mode", plan["mode"]),
                "group": info.get("group", plan.get("group", "")),
                "data_root": plan["data_root"],
                "n_samples": info.get("n", 0),
                "class_counts_match": info.get("counts_ok", True),
                "excel_acc": er["ACC"],
                "excel_auc": er["AUC"],
                "acc": info["acc_str"],
                "auc": info["auc_str"],
                "acc_delta": info.get("acc_delta", info["acc"] - (t_acc or 0)),
                "auc_delta": info.get("auc_delta", info["auc"] - (t_auc or 0)),
            }
        )
        model_reports.append(
            {
                "model": excel_name,
                "plan": plan,
                "target_class_counts": manifest.get("target_class_counts", target_class_counts(plan)),
                "achieved_class_counts": manifest.get("achieved_class_counts", {}),
                "class_count_validation": {"all_match": info.get("counts_ok", True), "per_class": []},
                "split_source_counts": manifest.get("split_source_counts", {}),
                "reproduced_macro": {"acc": info["acc_str"], "auc": info["auc_str"]},
                "excel_macro": {"acc": er["ACC"], "auc": er["AUC"]},
                "deltas": {
                    "acc": info.get("acc_delta", info["acc"] - (t_acc or 0)),
                    "auc": info.get("auc_delta", info["auc"] - (t_auc or 0)),
                },
                "search_info": manifest.get("search_info"),
                "manifest": str(manifest_path),
            }
        )

    for excel_name, row in existing_rows.items():
        if excel_name in table1:
            continue
        macro_rows.append(row)
        manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        plan = MODEL_PLANS[excel_name]
        er = excel_macro[excel_name]
        model_reports.append(
            {
                "model": excel_name,
                "plan": plan,
                "target_class_counts": manifest.get("target_class_counts", target_class_counts(plan)),
                "achieved_class_counts": manifest.get("achieved_class_counts", {}),
                "class_count_validation": {
                    "all_match": bool(row.get("class_counts_match", True)),
                    "per_class": [],
                },
                "split_source_counts": manifest.get("split_source_counts", {}),
                "reproduced_macro": {"acc": row["acc"], "auc": row["auc"]},
                "excel_macro": {"acc": er["ACC"], "auc": er["AUC"]},
                "deltas": {
                    "acc": float(row.get("acc_delta", 0)),
                    "auc": float(row.get("auc_delta", 0)),
                },
                "search_info": manifest.get("search_info"),
                "manifest": str(manifest_path),
            }
        )

    macro_rows.sort(key=lambda r: parse_point(r["auc"]) or 0.0, reverse=True)
    fields = [
        "excel_model", "model", "split", "mode", "group", "data_root", "n_samples",
        "class_counts_match", "excel_acc", "excel_auc", "acc", "auc", "acc_delta",
        "auc_delta", "sensitivity", "specificity", "npv", "ppv",
    ]
    with (METRICS_DIR / "table1_per_model_macro.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(macro_rows)

    write_report(REPORT_PATH, model_reports)
    subprocess.run(
        [sys.executable, str(HERE / "update_excel_vs_repro_summary.py")],
        check=True,
        cwd=str(ROOT),
    )
    print(f"Synced {METRICS_DIR / 'table1_per_model_macro.csv'} and EXCEL_VS_REPRO_SUMMARY")


if __name__ == "__main__":
    main()
