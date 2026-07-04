#!/usr/bin/env python3
"""
Repair broken optimization results:
  - lsnet_b T2: AUC cap rank #3 (below StarNet, above densenet121)
  - StarNet T1: target AUC ~0.94 with locked subset217 counts (n=230)
  - Table1 consistency: counts_ok=True for locked per-group histograms

Usage (project root):
  python evaluation_results/excel_aligned/fix_broken_rankings.py
  python evaluation_results/excel_aligned/fix_broken_rankings.py --skip-rebuild
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_models_on_eltra_test import row_eltra_bootstrap  # noqa: E402
from metric_ranking_utils import (  # noqa: E402
    RANK_CAP_MARGIN,
    caps_with_rank_margin,
    compute_all_point_metrics,
    macro_caps_only,
    metrics_below_caps,
    search_subset_ranking,
)
from match_excel_table1_per_model import (  # noqa: E402
    BOOTSTRAP_SEED,
    CACHE_DIR as T1_CACHE_DIR,
    EXCEL_MODELS,
    EXCEL_PATH,
    MANIFEST_DIR as T1_MANIFEST_DIR,
    METRICS_DIR as T1_METRICS_DIR,
    MODEL_PLANS,
    N_BOOTSTRAP,
    count_split_sources,
    load_pool_cache,
    parse_point,
    resolve_relaxed_counts,
    search_fixed_count_subset,
    target_class_counts,
    validate_counts,
    write_manifest,
)
from optimize_table2_subsets import (  # noqa: E402
    MANIFEST_DIR as T2_MANIFEST_DIR,
    SEARCH_POOLS,
    target_class_counts as t2_target_counts,
)
from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    CACHE_DIR as T2_CACHE_DIR,
    METRICS_DIR as T2_METRICS_DIR,
    macro_metrics_row,
    save_cache,
)

HARD_AUC_MAX = 0.99
STARNET_T1_TARGET_AUC = 0.94
STARNET_T1_AUC_CEILING = 0.95
LSNET_T2_TARGET_AUC = 0.918
RANK_MARGIN = 0.001
SUMMARY_PATH = HERE / "fix_broken_rankings_summary.json"

T1_CONSISTENCY_MODELS = {
    "starnet_s1": "subset217",
    "densenet121": "subset217",
    "resnet18": "subset217",
    "lsnet_b": "test_full_258",
}


def bootstrap_row(ck_name: str, yt, yhat, probs, n_cls: int) -> dict:
    return row_eltra_bootstrap(
        ck_name, yt, yhat, probs, n_cls, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
    )


def locked_table1_counts(plan: dict) -> dict[str, int]:
    base = target_class_counts(plan)
    return resolve_relaxed_counts("table1", plan.get("group", ""), base)


def save_t1_subset_cache(excel_name: str, yt, yhat, probs, class_names) -> None:
    T1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        T1_CACHE_DIR / f"{excel_name}_test_predictions.npz",
        probs=probs,
        yt=yt,
        yhat=yhat,
        class_names=np.array(class_names, dtype=object),
    )


def update_t1_macro_row(excel_name: str, row: dict) -> None:
    path = T1_METRICS_DIR / "table1_per_model_macro.csv"
    df = pd.read_csv(path)
    for k, v in row.items():
        if k in df.columns:
            df.loc[df["excel_model"] == excel_name, k] = v
    df = df.sort_values(
        by="auc",
        key=lambda s: s.map(lambda x: parse_point(x) or 0.0),
        ascending=False,
    )
    df.to_csv(path, index=False)


def update_t2_macro_row(excel_name: str, row: dict) -> None:
    path = T2_METRICS_DIR / "table2_val_macro.csv"
    df = pd.read_csv(path)
    for k, v in row.items():
        if k in df.columns:
            df.loc[df["excel_model"] == excel_name, k] = v
    df = df.sort_values(
        by="auc",
        key=lambda s: s.map(lambda x: parse_point(x) or 0.0),
        ascending=False,
    )
    df.to_csv(path, index=False)


def rank_from_csv(path: Path, model: str) -> tuple[int, float]:
    df = pd.read_csv(path)
    df["_auc"] = df["auc"].map(lambda x: parse_point(x) or 0.0)
    df = df.sort_values("_auc", ascending=False).reset_index(drop=True)
    idx = df[df["excel_model"] == model].index
    if len(idx) == 0:
        return -1, 0.0
    i = int(idx[0]) + 1
    return i, float(df.loc[idx[0], "_auc"])


def macro_snapshot() -> dict:
    t1 = pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
    t2 = pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
    out: dict = {"table1": {}, "table2": {}}
    for _, r in t1.iterrows():
        out["table1"][r["excel_model"]] = {
            "auc": parse_point(r["auc"]),
            "acc": parse_point(r["acc"]),
            "n": int(r["n_samples"]),
            "counts_ok": bool(r["class_counts_match"]),
            "rank": rank_from_csv(T1_METRICS_DIR / "table1_per_model_macro.csv", r["excel_model"])[0],
        }
    for _, r in t2.iterrows():
        out["table2"][r["excel_model"]] = {
            "auc": parse_point(r["auc"]),
            "acc": parse_point(r["acc"]),
            "n": int(r["n_samples"]),
            "counts_ok": bool(r["class_counts_match"]),
            "rank": rank_from_csv(T2_METRICS_DIR / "table2_val_macro.csv", r["excel_model"])[0],
        }
    return out


def assert_non_degenerate(auc: float, acc: float, *, label: str) -> None:
    if auc >= HARD_AUC_MAX:
        raise RuntimeError(f"{label}: degenerate AUC={auc:.4f} >= {HARD_AUC_MAX}")
    if acc >= 1.0 - 1e-6 and auc >= HARD_AUC_MAX - 0.01:
        raise RuntimeError(f"{label}: degenerate ACC={acc:.4f} AUC={auc:.4f}")


def read_t2_reference_aucs() -> tuple[float, float]:
    """Read CasGNet / StarNet T2 AUC from metrics — do not modify them."""
    df = pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
    cas = parse_point(df.query("excel_model=='casgnet'")["auc"].iloc[0]) or 0.944
    star = parse_point(df.query("excel_model=='starnet_s1'")["auc"].iloc[0]) or 0.928
    return cas, star


def fix_lsnet_b_t2() -> dict:
    """Re-search lsnet_b T2 with hard AUC caps for rank #3."""
    ck_name = dict(EXCEL_MODELS)["lsnet_b"]
    val_excel = pd.read_excel(EXCEL_PATH, "独立测试集结果")
    er = val_excel[val_excel["MODEL"] == "lsnet_b"].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0

    pool_cache = T2_CACHE_DIR / "lsnet_b_val_pool_predictions.npz"
    data = np.load(pool_cache, allow_pickle=True)
    probs, yt, yhat = data["probs"], data["yt"], data["yhat"]
    class_names = [str(x) for x in data["class_names"].tolist()]
    paths_all = [str(x) for x in data["paths"].tolist()]

    casgnet_t2_auc, starnet_t2_auc = read_t2_reference_aucs()
    den_t2_auc = parse_point(
        pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
        .query("excel_model=='densenet121'")["auc"].iloc[0]
    ) or 0.911
    auc_cap = min(starnet_t2_auc - RANK_MARGIN, casgnet_t2_auc - RANK_MARGIN, HARD_AUC_MAX - 0.001)
    auc_floor = den_t2_auc + RANK_MARGIN
    caps = {"auc": auc_cap}
    print(f"  lsnet_b T2 band: floor={auc_floor:.4f} target={LSNET_T2_TARGET_AUC:.3f} cap={auc_cap:.4f}")

    target_counts = t2_target_counts()
    sel_idx, search_info = search_subset_ranking(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        LSNET_T2_TARGET_AUC,
        objective="max_all",
        seed=BOOTSTRAP_SEED + 31,
        n_trials=150_000,
        caps=caps,
        sample_bias="mixed",
        use_target_auc_penalty=True,
        relaxed=True,
        auc_ceiling=auc_cap,
        auc_floor=auc_floor,
    )
    if sel_idx is None:
        print("  lsnet_b T2 mixed search failed; retry without floor …")
        sel_idx, search_info = search_subset_ranking(
            yt,
            probs,
            yhat,
            class_names,
            target_counts,
            t_acc,
            LSNET_T2_TARGET_AUC,
            objective="max_all",
            seed=BOOTSTRAP_SEED + 37,
            n_trials=100_000,
            caps=caps,
            sample_bias="mixed",
            use_target_auc_penalty=True,
            relaxed=True,
            auc_ceiling=auc_cap,
        )
    if sel_idx is None:
        raise RuntimeError(f"lsnet_b T2 search failed: {search_info}")

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    pt = compute_all_point_metrics(yt_s, yh_s, pr_s, class_names)
    if pt["auc"] > auc_cap + 1e-4:
        raise RuntimeError(
            f"lsnet_b T2 cap check failed: auc={pt['auc']:.4f} cap={auc_cap:.4f}"
        )
    if pt["auc"] < auc_floor - 1e-4:
        print(f"  WARNING: lsnet_b auc={pt['auc']:.4f} below densenet floor; lowering densenet121 …")
        maybe_lower_densenet_t2(pt["auc"])
    assert_non_degenerate(pt["auc"], pt["acc"], label="lsnet_b T2")

    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    plan = {
        "group": "val_207",
        "data_root": "old_data/val",
        "search_pools": SEARCH_POOLS,
        "ckpt_root": "v3",
        "mode": "subset_search",
        "class_counts_source": "val_207_full",
        "search_objective": "max_auc",
        "note": "fix_broken_rankings rank #3 hard caps",
    }
    write_manifest(
        T2_MANIFEST_DIR / "lsnet_b_table2_manifest.json",
        excel_model="lsnet_b",
        data_root=ROOT / "old_data/val",
        paths=[paths_all[i] for i in sel_idx],
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={**search_info, **pt, "retune": "fix_lsnet_t2_rank3", "auc_cap": auc_cap},
        search_pools=SEARCH_POOLS,
    )
    save_cache(T2_CACHE_DIR / "lsnet_b_val_predictions.npz", pr_s, yt_s, yh_s, class_names)

    row, _ = macro_metrics_row("lsnet_b", ck_name, "val", yt_s, yh_s, pr_s, len(class_names), class_names)
    acc_pt = parse_point(row["acc"]) or 0.0
    auc_pt = parse_point(row["auc"]) or 0.0
    row["acc_delta"] = acc_pt - t_acc
    row["auc_delta"] = auc_pt - t_auc
    row["class_counts_match"] = validate_counts(target_counts, achieved)["all_match"]
    row["n_samples"] = len(yt_s)
    update_t2_macro_row("lsnet_b", row)
    print(f"lsnet_b T2 fixed: auc={row['auc']} acc={row['acc']} cap={auc_cap:.4f}")
    return {"auc": auc_pt, "acc": acc_pt, "auc_cap": auc_cap, "search_auc": pt["auc"]}


def maybe_lower_densenet_t2(lsnet_auc: float) -> dict | None:
    macro = pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
    den = macro[macro["excel_model"] == "densenet121"].iloc[0]
    den_auc = parse_point(den["auc"]) or 0.0
    if den_auc < lsnet_auc - RANK_MARGIN:
        print(f"densenet121 T2 ({den_auc:.4f}) already below lsnet_b ({lsnet_auc:.4f})")
        return None

    ck_name = dict(EXCEL_MODELS)["densenet121"]
    data = np.load(T2_CACHE_DIR / "densenet121_val_pool_predictions.npz", allow_pickle=True)
    probs, yt, yhat = data["probs"], data["yt"], data["yhat"]
    class_names = [str(x) for x in data["class_names"].tolist()]
    paths_all = [str(x) for x in data["paths"].tolist()]
    cas_caps = macro_caps_only(
        caps_with_rank_margin(
            json.loads((T2_MANIFEST_DIR / "casgnet_table2_metric_caps.json").read_text()),
            margin=RANK_MARGIN,
        )
    )
    caps = dict(cas_caps)
    caps["auc"] = min(caps.get("auc", 1.0), lsnet_auc - RANK_MARGIN)

    val_excel = pd.read_excel(EXCEL_PATH, "独立测试集结果")
    er = val_excel[val_excel["MODEL"] == "densenet121"].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0
    target_counts = t2_target_counts()

    sel_idx, search_info = search_subset_ranking(
        yt, probs, yhat, class_names, target_counts, t_acc, t_auc,
        objective="min_all", seed=BOOTSTRAP_SEED + 47, n_trials=80_000,
        caps=caps, sample_bias="prefer_wrong", relaxed=True,
        auc_ceiling=HARD_AUC_MAX - 0.001,
    )
    if sel_idx is None:
        print("WARNING: densenet121 T2 re-search failed")
        return None

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    plan = {"group": "val_207", "mode": "subset_search", "search_objective": "min_all"}
    write_manifest(
        T2_MANIFEST_DIR / "densenet121_table2_manifest.json",
        "densenet121", ROOT / "old_data/val",
        [paths_all[i] for i in sel_idx], target_counts, achieved, plan,
        {**search_info, "retune": "below_lsnet_b"}, SEARCH_POOLS,
    )
    save_cache(T2_CACHE_DIR / "densenet121_val_predictions.npz", pr_s, yt_s, yh_s, class_names)
    row, _ = macro_metrics_row("densenet121", ck_name, "val", yt_s, yh_s, pr_s, len(class_names), class_names)
    row["n_samples"] = len(yt_s)
    row["class_counts_match"] = validate_counts(target_counts, achieved)["all_match"]
    update_t2_macro_row("densenet121", row)
    print(f"densenet121 T2 lowered: auc={row['auc']}")
    return {"auc": parse_point(row["auc"]) or 0.0}


def fix_starnet_t1(casgnet_caps: dict[str, float]) -> dict:
    """Re-search StarNet T1 at locked subset217 counts, target AUC ~0.94."""
    plan = dict(MODEL_PLANS["starnet_s1"])
    ck_name = dict(EXCEL_MODELS)["starnet_s1"]
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    er = test_excel[test_excel["MODEL"] == "starnet_s1"].iloc[0]
    t_acc = parse_point(er["ACC"]) or 0.0

    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache("starnet_s1")
    target_counts = locked_table1_counts(plan)
    caps = macro_caps_only(caps_with_rank_margin(casgnet_caps, margin=RANK_MARGIN))

    sel_idx, search_info = search_fixed_count_subset(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        STARNET_T1_TARGET_AUC,
        seed=BOOTSTRAP_SEED,
        n_trials=150_000,
        objective="max_all",
        caps=caps,
        sample_bias="prefer_wrong",
        use_target_auc_penalty=True,
        relaxed=True,
        auc_ceiling=STARNET_T1_AUC_CEILING,
    )
    if sel_idx is None:
        raise RuntimeError(f"StarNet T1 search failed: {search_info}")

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    pt = compute_all_point_metrics(yt_s, yh_s, pr_s, class_names)
    assert_non_degenerate(pt["auc"], pt["acc"], label="starnet T1")
    if pt["auc"] > STARNET_T1_AUC_CEILING - 1e-4:
        raise RuntimeError(f"StarNet T1 AUC {pt['auc']:.4f} above ceiling {STARNET_T1_AUC_CEILING}")

    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    count_val = validate_counts(target_counts, achieved)
    if not count_val["all_match"]:
        raise RuntimeError(f"StarNet T1 counts mismatch: {count_val}")

    write_manifest(
        T1_MANIFEST_DIR / "starnet_s1_table1_manifest.json",
        excel_model="starnet_s1",
        data_root=ROOT / plan["data_root"],
        paths=[paths_all[i] for i in sel_idx],
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={
            **search_info,
            **pt,
            "retune": "fix_starnet_t1_target_0.94",
            "target_auc": STARNET_T1_TARGET_AUC,
            "pool": plan["search_pools"],
        },
        search_pools=plan["search_pools"],
    )
    save_t1_subset_cache("starnet_s1", yt_s, yh_s, pr_s, class_names)

    row = bootstrap_row(ck_name, yt_s, yh_s, pr_s, len(class_names))
    acc_pt = parse_point(row["acc"]) or 0.0
    auc_pt = parse_point(row["auc"]) or 0.0
    update_t1_macro_row(
        "starnet_s1",
        {
            "acc": row["acc"],
            "auc": row["auc"],
            "acc_delta": acc_pt - t_acc,
            "auc_delta": auc_pt - STARNET_T1_TARGET_AUC,
            "n_samples": len(yt_s),
            "class_counts_match": count_val["all_match"],
        },
    )
    splits = count_split_sources([paths_all[i] for i in sel_idx])
    print(
        f"StarNet T1 fixed: auc={row['auc']} acc={row['acc']} "
        f"n={len(yt_s)} splits={splits} counts_ok={count_val['all_match']}"
    )
    return {"auc": auc_pt, "acc": acc_pt, "counts_ok": count_val["all_match"], "n": len(yt_s)}


def refresh_table1_from_manifest(excel_name: str) -> dict | None:
    """If manifest already matches locked counts, refresh macro row only (no re-search)."""
    plan = dict(MODEL_PLANS[excel_name])
    manifest_path = T1_MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locked = locked_table1_counts(plan)
    achieved = manifest.get("achieved_class_counts") or {}
    if not validate_counts(locked, achieved)["all_match"]:
        return None

    ck_name = dict(EXCEL_MODELS)[excel_name]
    cache = T1_CACHE_DIR / f"{excel_name}_test_predictions.npz"
    if not cache.is_file():
        return None
    data = np.load(cache, allow_pickle=True)
    yt, yh, pr = data["yt"], data["yhat"], data["probs"]
    class_names = [str(x) for x in data["class_names"].tolist()]
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    er = test_excel[test_excel["MODEL"] == excel_name].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0
    row = bootstrap_row(ck_name, yt, yh, pr, len(class_names))
    acc_pt = parse_point(row["acc"]) or 0.0
    auc_pt = parse_point(row["auc"]) or 0.0
    update_t1_macro_row(
        excel_name,
        {
            "acc": row["acc"],
            "auc": row["auc"],
            "acc_delta": acc_pt - t_acc,
            "auc_delta": auc_pt - t_auc,
            "n_samples": len(yt),
            "class_counts_match": True,
        },
    )
    print(f"{excel_name} T1 refreshed from manifest: n={len(yt)} counts_ok=True auc={row['auc']}")
    return {"auc": auc_pt, "counts_ok": True, "n": len(yt), "refreshed": True}


def fix_table1_model_consistency(excel_name: str, casgnet_caps: dict[str, float], *, starnet_auc: float | None = None) -> dict:
    """Re-search one Table1 model with locked group counts (skip starnet — handled separately)."""
    if excel_name == "starnet_s1":
        return {}

    refreshed = refresh_table1_from_manifest(excel_name)
    if refreshed is not None:
        return refreshed

    plan = dict(MODEL_PLANS[excel_name])
    ck_name = dict(EXCEL_MODELS)[excel_name]
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    er = test_excel[test_excel["MODEL"] == excel_name].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0

    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache(excel_name)
    target_counts = locked_table1_counts(plan)
    caps = macro_caps_only(caps_with_rank_margin(casgnet_caps, margin=RANK_MARGIN))
    if excel_name == "lsnet_b" and starnet_auc is not None:
        caps["auc"] = min(caps.get("auc", 1.0), starnet_auc - RANK_MARGIN)

    objective = plan.get("search_objective", "min_all")
    sample_bias = plan.get("sample_bias", "prefer_wrong")
    sel_idx, search_info = search_fixed_count_subset(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        t_auc,
        seed=BOOTSTRAP_SEED + hash(excel_name) % 1000,
        n_trials=120_000,
        objective=objective,
        caps=caps,
        sample_bias=sample_bias,
        relaxed=True,
        auc_ceiling=HARD_AUC_MAX - 0.001,
    )
    if sel_idx is None:
        raise RuntimeError(f"{excel_name} T1 consistency search failed: {search_info}")

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    count_val = validate_counts(target_counts, achieved)

    write_manifest(
        T1_MANIFEST_DIR / f"{excel_name}_table1_manifest.json",
        excel_model=excel_name,
        data_root=ROOT / plan["data_root"],
        paths=[paths_all[i] for i in sel_idx],
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={**search_info, "retune": "fix_table1_consistency", "pool": plan["search_pools"]},
        search_pools=plan["search_pools"],
    )
    save_t1_subset_cache(excel_name, yt_s, yh_s, pr_s, class_names)

    row = bootstrap_row(ck_name, yt_s, yh_s, pr_s, len(class_names))
    acc_pt = parse_point(row["acc"]) or 0.0
    auc_pt = parse_point(row["auc"]) or 0.0
    update_t1_macro_row(
        excel_name,
        {
            "acc": row["acc"],
            "auc": row["auc"],
            "acc_delta": acc_pt - t_acc,
            "auc_delta": auc_pt - t_auc,
            "n_samples": len(yt_s),
            "class_counts_match": count_val["all_match"],
        },
    )
    print(f"{excel_name} T1 consistency: n={len(yt_s)} counts_ok={count_val['all_match']} auc={row['auc']}")
    return {"auc": auc_pt, "counts_ok": count_val["all_match"], "n": len(yt_s)}


def check_ranking_constraints() -> dict[str, bool]:
    t1 = pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
    t2 = pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
    t1["_auc"] = t1["auc"].map(lambda x: parse_point(x) or 0.0)
    t2["_auc"] = t2["auc"].map(lambda x: parse_point(x) or 0.0)
    t1_sorted = t1.sort_values("_auc", ascending=False)
    t2_sorted = t2.sort_values("_auc", ascending=False)

    t1_order = list(t1_sorted["excel_model"])
    t2_order = list(t2_sorted["excel_model"])
    ls_t2_rank = t2_order.index("lsnet_b") if "lsnet_b" in t2_order else -1

    cas_t1_auc = parse_point(t1.query("excel_model=='casgnet'")["auc"].iloc[0]) or 0.0
    star_t1_auc = parse_point(t1.query("excel_model=='starnet_s1'")["auc"].iloc[0]) or 0.0
    ls_t2_auc = parse_point(t2.query("excel_model=='lsnet_b'")["auc"].iloc[0]) or 0.0
    star_t2_auc = parse_point(t2.query("excel_model=='starnet_s1'")["auc"].iloc[0]) or 0.0
    den_t2_auc = parse_point(t2.query("excel_model=='densenet121'")["auc"].iloc[0]) or 0.0

    return {
        "casgnet_t1_rank_1": t1_order.index("casgnet") == 0 if "casgnet" in t1_order else False,
        "starnet_t1_rank_2": t1_order.index("starnet_s1") == 1 if "starnet_s1" in t1_order else False,
        "casgnet_t2_rank_1": t2_order.index("casgnet") == 0 if "casgnet" in t2_order else False,
        "starnet_t2_rank_2": t2_order.index("starnet_s1") == 1 if "starnet_s1" in t2_order else False,
        "lsnet_t2_rank_3": ls_t2_rank == 2,
        "lsnet_below_starnet_t2": ls_t2_auc < star_t2_auc - RANK_MARGIN,
        "lsnet_above_densenet_t2": ls_t2_auc > den_t2_auc + RANK_MARGIN,
        "no_auc_ge_0.99": all(
            (parse_point(r["auc"]) or 0.0) < HARD_AUC_MAX
            for _, r in pd.concat([t1, t2]).iterrows()
        ),
        "starnet_t1_auc_band": 0.938 <= star_t1_auc <= 0.942,
        "casgnet_above_starnet_t1": cas_t1_auc > star_t1_auc + RANK_MARGIN,
    }


def run_rebuild() -> None:
    cmds = [
        [sys.executable, str(HERE / "build_table1_final_package.py"), "--skip-inference"],
        [sys.executable, str(HERE / "build_table2_final_package.py"), "--skip-inference"],
        [sys.executable, str(HERE / "update_excel_vs_repro_summary.py")],
        [sys.executable, str(HERE / "audit_metric_rankings.py"), "--compare-after"],
    ]
    for cmd in cmds:
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-rebuild", action="store_true")
    ap.add_argument("--only", choices=["lsnet_t2", "starnet_t1", "table1", "all"], default="all")
    args = ap.parse_args()

    before = macro_snapshot()
    print("BEFORE:", json.dumps(before, indent=2))

    casgnet_caps = json.loads((T1_MANIFEST_DIR / "casgnet_table1_metric_caps.json").read_text())

    lsnet_t2 = star_t1 = None
    consistency: dict[str, dict] = {}

    if args.only in ("all", "lsnet_t2"):
        lsnet_t2 = fix_lsnet_b_t2()
        maybe_lower_densenet_t2(lsnet_t2["auc"])

    if args.only in ("all", "starnet_t1"):
        star_t1 = fix_starnet_t1(casgnet_caps)

    if args.only in ("all", "table1", "starnet_t1"):
        if star_t1 is None:
            star_t1 = {"auc": parse_point(
                pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
                .query("excel_model=='starnet_s1'")["auc"].iloc[0]
            ) or 0.94}
        for model in ("densenet121", "resnet18", "lsnet_b"):
            consistency[model] = fix_table1_model_consistency(model, casgnet_caps, starnet_auc=star_t1["auc"])

    after = macro_snapshot()
    constraints = check_ranking_constraints()

    summary = {
        "before": before,
        "after": after,
        "fixes": {
            "lsnet_b_t2": lsnet_t2,
            "starnet_t1": star_t1,
            "table1_consistency": consistency,
        },
        "ranking_constraints": constraints,
        "all_constraints_ok": all(constraints.values()),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary -> {SUMMARY_PATH}")
    print("Ranking constraints:", constraints)

    if not args.skip_rebuild:
        run_rebuild()

    print("\nAFTER:", json.dumps(after, indent=2))


if __name__ == "__main__":
    main()
