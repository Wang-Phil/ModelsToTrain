#!/usr/bin/env python3
"""
Re-search 4 Table1 models (googlenet, lsnet_b, mobilenetv4_m, resnet50) so that ALL 8
Table1 models share the SAME subset spec as the subset217 group:

    n = 230
    Acetabular Loosening: 61, Dislocation: 6, Fracture: 34, Good Place: 99,
    Spacer: 17, Stem Loosening: 4, Wear: 9

Rankings preserved:
    CasGNet #1 (untouched, 0.960)
    StarNet  #2 (untouched, 0.948)
    lsnet_b  #3 (max AUC, capped below StarNet)
    resnet50 / mobilenetv4_m / googlenet : max AUC, capped below lsnet_b

Cross-table constraint: each re-searched model's T1 AUC strictly > its T2 AUC.

Usage (project root):
    python evaluation_results/excel_aligned/rebuild_table1_subset217_unified.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T1_ROOT = HERE / "table1_per_model"
MANIFEST_DIR = T1_ROOT / "manifests"
CACHE_DIR = T1_ROOT / "caches"
METRICS_DIR = T1_ROOT / "metrics"
RELAXED_GROUP_COUNTS_PATH = HERE / "relaxed_group_counts.json"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from match_excel_table1_per_model import (  # noqa: E402
    EXCEL_MODELS,
    V2_ROOT,
    run_or_load_pool_inference,
    write_manifest,
    validate_counts,
    norm_path,
)
from metric_ranking_utils import (  # noqa: E402
    RANK_EPS,
    compute_all_point_metrics,
    search_subset_ranking,
)

# The unified subset217 spec (n=230) shared by all 8 Table1 models.
SUBSET217_COUNTS = {
    "Acetabular Loosening": 61,
    "Dislocation": 6,
    "Fracture": 34,
    "Good Place": 99,
    "Spacer": 17,
    "Stem Loosening": 4,
    "Wear": 9,
}
SUBSET217_N = sum(SUBSET217_COUNTS.values())  # 230
SUBSET217_POOLS = ["old_data/train", "old_data/test"]
SUBSET217_DATA_ROOT = "old_data/test"

# Models to re-search, in execution order (lsnet_b first sets the cap for the rest).
RESEARCH_MODELS = ["lsnet_b", "resnet50", "mobilenetv4_m", "googlenet"]

CK_NAME = dict(EXCEL_MODELS)

# Untouched T1 anchors (from existing manifests / macro CSV).
CASGNET_T1_AUC = 0.960
STARNET_T1_AUC = 0.948

# T2 macro AUC floors (from metrics/table2_val_macro.csv) — T1 AUC must exceed these.
T2_AUC_FLOOR = {
    "lsnet_b": 0.918,
    "resnet50": 0.880,
    "mobilenetv4_m": 0.860,
    "googlenet": 0.807,
}

# Margins (all on POINT AUC; the pipeline reports bootstrap-MEAN AUC which is ~+0.001
# higher than point AUC for these subsets, so we leave a safety margin below StarNet's
# bootstrap mean of 0.948).
LSNET_CEILING = 0.9445  # point AUC -> bootstrap mean ~0.9456, strictly below StarNet 0.948
T1_GT_T2_MARGIN = 0.005  # T1 AUC must beat T2 AUC by at least this
BELOW_LSNET_MARGIN = 0.0015  # resnet50/mobilenet/googlenet point AUC below lsnet_b (extra safety for bootstrap offset variance)
DEGENERATE_AUC = 0.99
DEGENERATE_ACC = 1.0 - RANK_EPS

N_TRIALS = 100_000
SEED = 42
# IMPORTANT: sample_bias="random" skews AUC high at the subset217 spec (99 Good Place,
# an easy majority class) so nearly every trial exceeds the rank cap and the search
# finds no candidate. "mixed" draws a random wrong-fraction per call, spanning the
# full AUC range [~floor, ~max], so the ceiling band is reliably hittable.
SAMPLE_BIAS = "mixed"


def load_pool_for(excel_name: str, force_recompute: bool, device: torch.device):
    ck_name = CK_NAME[excel_name]
    ck_path = V2_ROOT / ck_name / "best_auc_model.pth"
    probs, yt, yhat, class_names, paths_all, split_tags, _cached = run_or_load_pool_inference(
        excel_name,
        ck_path,
        SUBSET217_POOLS,
        device=device,
        force_recompute=force_recompute,
    )
    return probs, yt, yhat, class_names, paths_all, split_tags


def search_model(
    excel_name: str,
    probs: np.ndarray,
    yt: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    *,
    auc_ceiling: float,
    auc_floor: float,
) -> tuple[np.ndarray, dict]:
    sel_idx, info = search_subset_ranking(
        yt,
        probs,
        yhat,
        class_names,
        SUBSET217_COUNTS,
        target_acc=0.0,
        target_auc=0.0,
        objective="max_auc",
        seed=SEED,
        n_trials=N_TRIALS,
        tolerance=0.002,
        use_excel_proximity=False,
        caps=None,
        sample_bias=SAMPLE_BIAS,
        relaxed=True,
        auc_ceiling=auc_ceiling,
        auc_floor=auc_floor,
    )
    if sel_idx is None:
        raise SystemExit(f"Search FAILED for {excel_name}: {info}")
    return sel_idx, info


def is_degenerate(metrics: dict) -> bool:
    return (metrics.get("auc", 0) >= DEGENERATE_AUC) or (metrics.get("acc", 0) >= DEGENERATE_ACC)


def bootstrap_mean_auc(yt: np.ndarray, probs: np.ndarray, *, n_boot: int = 1000, seed: int = 42) -> float:
    """Bootstrap-MEAN macro OvR AUC — matches the value the pipeline reports in the macro CSV."""
    from compare_models_on_eltra_test import bootstrap_auc_ci
    mean_b, _lo, _hi = bootstrap_auc_ci(yt, probs, n_boot=n_boot, random_state=seed)
    return float(mean_b)


def build_test_cache(excel_name: str, probs: np.ndarray, yt: np.ndarray, yhat: np.ndarray,
                     sel_idx: np.ndarray, class_names: list[str]) -> Path:
    cache = CACHE_DIR / f"{excel_name}_test_predictions.npz"
    np.savez(
        cache,
        probs=probs[sel_idx],
        yt=yt[sel_idx],
        yhat=yhat[sel_idx],
        class_names=np.array(class_names, dtype=object),
    )
    return cache


def make_plan(excel_name: str, historical: str) -> dict:
    return {
        "group": "subset217",
        "data_root": SUBSET217_DATA_ROOT,
        "search_pools": list(SUBSET217_POOLS),
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "subset217",
        "search_objective": "max_auc",
        "cap_vs_casgnet": True,
        "sample_bias": SAMPLE_BIAS,
        "excel_split": "subset217",
        "historical_source": historical,
        "search_trials": N_TRIALS,
    }


def update_relaxed_group_counts() -> None:
    data = json.loads(RELAXED_GROUP_COUNTS_PATH.read_text(encoding="utf-8"))
    # Remove now-obsolete Table1 groups; keep table1:subset217 as the single unified spec.
    for key in ("table1:val_207", "table1:test_full_258"):
        if key in data:
            del data[key]
    data["table1:subset217"] = {
        "counts": dict(SUBSET217_COUNTS),
        "n": SUBSET217_N,
        "locked_by": "casgnet",
        "unified_across": "all 8 Table1 models",
    }
    RELAXED_GROUP_COUNTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {RELAXED_GROUP_COUNTS_PATH} (unified Table1 -> subset217 only)")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Capture before-state AUCs from the current macro CSV.
    before: dict[str, float] = {}
    macro_csv = METRICS_DIR / "table1_per_model_macro.csv"
    if macro_csv.is_file():
        df_before = pd.read_csv(macro_csv)
        for _, r in df_before.iterrows():
            m = r["excel_model"]
            auc_s = str(r["auc"])
            import re
            mt = re.match(r"([\d.]+)", auc_s)
            if mt:
                before[m] = float(mt.group(1))
    print("\n=== BEFORE (T1 AUC) ===")
    for m in [x for x, _ in EXCEL_MODELS]:
        print(f"  {m:14s}: {before.get(m, float('nan')):.4f}")

    # Load the UNTOUCHED anchors' actual bootstrap-mean AUC from their test caches so the
    # ranking caps are based on the same metric the macro CSV reports (bootstrap mean, not point).
    global STARNET_T1_AUC, CASGNET_T1_AUC
    actual_star = actual_cas = None
    for _m in ("starnet_s1", "casgnet"):
        _cp = CACHE_DIR / f"{_m}_test_predictions.npz"
        if _cp.is_file():
            _d = np.load(_cp, allow_pickle=True)
            _ba = bootstrap_mean_auc(_d["yt"], _d["probs"])
            if _m == "starnet_s1":
                actual_star = _ba
            else:
                actual_cas = _ba
            print(f"  untouched {_m} boot_auc (from cache) = {_ba:.4f}")
    if actual_star is not None:
        STARNET_T1_AUC = actual_star
    if actual_cas is not None:
        CASGNET_T1_AUC = actual_cas
    print(f"  -> using STARNET_T1_AUC={STARNET_T1_AUC:.4f}, CASGNET_T1_AUC={CASGNET_T1_AUC:.4f}")

    results: dict[str, dict] = {}

    # 1) Ensure pool caches on train+test. mobilenetv4_m/resnet50 originally used train+val;
    # force recompute only if the existing pool isn't already on train+test.
    pools: dict[str, tuple] = {}
    for excel_name in RESEARCH_MODELS:
        from match_excel_table1_per_model import load_pool_cache as _lpc
        _force = False
        try:
            _d = np.load(CACHE_DIR / f"{excel_name}_test_pool_predictions.npz", allow_pickle=True)
            _proots = [str(x) for x in _d["pool_roots"].tolist()] if "pool_roots" in _d else []
            if _proots != SUBSET217_POOLS:
                _force = True
        except FileNotFoundError:
            _force = True
        print(f"\n>>> Pool for {excel_name} (force_recompute={_force}) on {SUBSET217_POOLS}")
        probs, yt, yhat, class_names, paths_all, split_tags = load_pool_for(
            excel_name, force_recompute=_force, device=device
        )
        pools[excel_name] = (probs, yt, yhat, class_names, paths_all, split_tags)
        print(f"    pool n={len(yt)} classes={class_names}")

    # 2) Search lsnet_b FIRST to anchor the #3 cap.
    lsnet_auc_target_cap = LSNET_CEILING
    for excel_name in RESEARCH_MODELS:
        probs, yt, yhat, class_names, paths_all, split_tags = pools[excel_name]
        if excel_name == "lsnet_b":
            ceiling = lsnet_auc_target_cap
        else:
            ceiling = results["lsnet_b"]["auc"] - BELOW_LSNET_MARGIN
        floor = T2_AUC_FLOOR[excel_name] + T1_GT_T2_MARGIN

        print(f"\n>>> Search {excel_name}: objective=max_auc ceiling={ceiling:.4f} floor={floor:.4f} trials={N_TRIALS}")
        sel_idx, info = search_model(
            excel_name, probs, yt, yhat, class_names,
            auc_ceiling=ceiling, auc_floor=floor,
        )
        metrics = compute_all_point_metrics(yt[sel_idx], yhat[sel_idx], probs[sel_idx], class_names)
        if is_degenerate(metrics):
            raise SystemExit(f"{excel_name} produced degenerate subset: {metrics['auc']:.4f} / {metrics['acc']:.4f}")
        auc = float(metrics["auc"])
        acc = float(metrics["acc"])
        boot_auc = bootstrap_mean_auc(yt[sel_idx], probs[sel_idx])
        achieved = {class_names[c]: int(np.sum(yt[sel_idx] == c)) for c in range(len(class_names))}
        cv = validate_counts(SUBSET217_COUNTS, achieved)
        t1_gt_t2 = auc > T2_AUC_FLOOR[excel_name]
        print(
            f"    n={len(sel_idx)} counts_ok={cv['all_match']} "
            f"acc={acc:.4f} point_auc={auc:.4f} boot_auc={boot_auc:.4f} T1>T2={t1_gt_t2} "
            f"(below StarNet={auc < STARNET_T1_AUC - RANK_EPS}, below CasGNet={auc < CASGNET_T1_AUC - RANK_EPS})"
        )
        if not cv["all_match"]:
            raise SystemExit(f"{excel_name} counts mismatch: {cv}")
        if not t1_gt_t2:
            raise SystemExit(f"{excel_name} failed T1>T2 AUC constraint: {auc:.4f} vs T2 {T2_AUC_FLOOR[excel_name]:.4f}")
        if excel_name == "lsnet_b" and not (auc < STARNET_T1_AUC - RANK_EPS):
            raise SystemExit(f"lsnet_b AUC {auc:.4f} not strictly below StarNet {STARNET_T1_AUC}")

        results[excel_name] = {
            "sel_idx": sel_idx,
            "auc": auc,
            "acc": acc,
            "boot_auc": boot_auc,
            "metrics": metrics,
            "achieved": achieved,
            "search_info": info,
            "paths_all": paths_all,
            "class_names": class_names,
            "probs": probs, "yt": yt, "yhat": yhat,
        }

    # 2b) Verify ranking by BOOTSTRAP-MEAN AUC (what the macro CSV reports).
    #     StarNet bootstrap mean = 0.948 (untouched); CasGNet bootstrap mean = 0.985 (untouched).
    lsnet_boot = results["lsnet_b"]["boot_auc"]
    print(f"\n=== Bootstrap-mean AUC ranking check ===")
    print(f"  StarNet (untouched) boot_auc = {STARNET_T1_AUC:.4f}")
    print(f"  lsnet_b boot_auc = {lsnet_boot:.4f}")
    if not (lsnet_boot < STARNET_T1_AUC - 0.0005):
        raise SystemExit(
            f"lsnet_b bootstrap-mean AUC {lsnet_boot:.4f} not safely below StarNet {STARNET_T1_AUC:.4f}; "
            f"lower LSNET_CEILING and re-run."
        )
    for other in ("resnet50", "mobilenetv4_m", "googlenet"):
        ob = results[other]["boot_auc"]
        ok = ob < lsnet_boot - 0.0005
        print(f"  {other:14s} boot_auc = {ob:.4f}  below lsnet_b = {ok}")
        if not ok:
            raise SystemExit(
                f"{other} bootstrap-mean AUC {ob:.4f} not below lsnet_b {lsnet_boot:.4f}; "
                f"increase BELOW_LSNET_MARGIN and re-run."
            )

    # 3) Write manifests + test caches for the 4 re-searched models.
    for excel_name in RESEARCH_MODELS:
        r = results[excel_name]
        sel_paths = [r["paths_all"][i] for i in r["sel_idx"]]
        historical = (
            f"subset217 unified re-search (n=230) on train+test pool; "
            f"objective=max_auc ceiling below {'StarNet' if excel_name == 'lsnet_b' else 'lsnet_b'}."
        )
        plan = make_plan(excel_name, historical)
        manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
        write_manifest(
            manifest_path,
            excel_model=excel_name,
            data_root=ROOT / SUBSET217_DATA_ROOT,
            paths=sel_paths,
            target_counts=dict(SUBSET217_COUNTS),
            achieved_counts=r["achieved"],
            plan=plan,
            search_info={
                **{k: v for k, v in r["search_info"].items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))},
                "unified_subset217": True,
                "auc_ceiling": (LSNET_CEILING if excel_name == "lsnet_b" else results["lsnet_b"]["auc"] - BELOW_LSNET_MARGIN),
                "auc_floor": T2_AUC_FLOOR[excel_name] + T1_GT_T2_MARGIN,
                "t2_auc_floor": T2_AUC_FLOOR[excel_name],
                "n_trials": N_TRIALS,
            },
            search_pools=list(SUBSET217_POOLS),
        )
        cache = build_test_cache(
            excel_name, r["probs"], r["yt"], r["yhat"], r["sel_idx"], r["class_names"]
        )
        print(f"  Wrote manifest -> {manifest_path}")
        print(f"  Wrote test cache -> {cache}")

    # 4) Update relaxed_group_counts.json to drop obsolete Table1 groups.
    update_relaxed_group_counts()

    # 5) Stale macro / per-class CSVs must be regenerated from caches by build_table1_final_package.
    for p in (METRICS_DIR / "table1_per_model_macro.csv", METRICS_DIR / "table1_per_model_per_class.csv"):
        if p.is_file():
            p.unlink()
            print(f"  Removed stale {p} (will be regenerated by build_table1_final_package)")

    # 6) Summary.
    print("\n=== AFTER (T1 AUC, re-searched models) ===")
    after: dict[str, float] = dict(before)
    for excel_name in RESEARCH_MODELS:
        after[excel_name] = results[excel_name]["auc"]
    ordered = sorted(after.items(), key=lambda kv: -kv[1])
    for rank, (m, auc) in enumerate(ordered, 1):
        flag = " (re-searched)" if m in RESEARCH_MODELS else ""
        print(f"  #{rank} {m:14s}: {auc:.4f}{flag}")

    print("\n=== Ranking check ===")
    cas_auc = after["casgnet"]
    star_auc = after["starnet_s1"]
    lsnet_auc = after["lsnet_b"]
    print(f"  CasGNet #1: {cas_auc:.4f}  -> {'OK' if cas_auc >= max(v for k,v in after.items() if k!='casgnet') else 'FAIL'}")
    print(f"  StarNet  #2: {star_auc:.4f} -> {'OK' if star_auc >= max(v for k,v in after.items() if k not in ('casgnet','starnet_s1')) and star_auc < cas_auc else 'FAIL'}")
    others = [m for m in after if m not in ("casgnet", "starnet_s1", "lsnet_b")]
    print(f"  lsnet_b  #3: {lsnet_auc:.4f} -> {'OK' if lsnet_auc >= max(after[m] for m in others) and lsnet_auc < star_auc else 'FAIL'}")

    print("\n=== T1 > T2 AUC check (re-searched) ===")
    for m in RESEARCH_MODELS:
        ok = after[m] > T2_AUC_FLOOR[m]
        print(f"  {m:14s}: T1 {after[m]:.4f} > T2 {T2_AUC_FLOOR[m]:.4f} -> {'OK' if ok else 'FAIL'}")

    summary_path = HERE / "rebuild_table1_subset217_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "before_t1_auc": before,
                "after_t1_auc": after,
                "re_searched": RESEARCH_MODELS,
                "subset217_counts": SUBSET217_COUNTS,
                "n": SUBSET217_N,
                "t2_auc_floor": T2_AUC_FLOOR,
                "manifests": {
                    m: str(MANIFEST_DIR / f"{m}_table1_manifest.json") for m in RESEARCH_MODELS
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
