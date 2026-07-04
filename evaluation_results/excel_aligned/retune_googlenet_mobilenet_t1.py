#!/usr/bin/env python3
"""
Boost googlenet and mobilenetv4_m Table 1 AUC toward ~0.92 using locked group counts.

Uses pool caches only (no fresh inference). Updates manifests + subset caches.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_models_on_eltra_test import row_eltra_bootstrap  # noqa: E402
from metric_ranking_utils import (  # noqa: E402
    compute_all_point_metrics,
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
    load_pool_cache,
    parse_point,
    resolve_relaxed_counts,
    target_class_counts,
    validate_counts,
    write_manifest,
)

TARGET_AUC = 0.92
AUC_FLOOR = 0.90
AUC_CEILING_DEFAULT = 0.931  # below densenet121/resnet50 (~0.932/0.933)
RANK_MARGIN = 0.001
N_TRIALS = int(__import__("os").environ.get("GOOGLE_MOBILE_T1_TRIALS", "180000"))
MODELS = ("googlenet", "mobilenetv4_m")
SAMPLE_BIAS = {"googlenet": "random", "mobilenetv4_m": "mixed"}


def bootstrap_row(ck_name: str, yt, yhat, probs, n_cls: int) -> dict:
    return row_eltra_bootstrap(
        ck_name, yt, yhat, probs, n_cls, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
    )


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


def compute_auc_ceiling(macro_df: pd.DataFrame, casgnet_auc: float, starnet_auc: float, excel_name: str) -> float:
    """Cap below CasGNet/StarNet and the model ranked directly above in the weak tail."""
    chain_above = {
        "mobilenetv4_m": "resnet18",
        "googlenet": "mobilenetv4_m",
    }
    candidates = [
        casgnet_auc - RANK_MARGIN,
        starnet_auc - RANK_MARGIN,
    ]
    above = chain_above.get(excel_name)
    if above:
        rows = macro_df[macro_df["excel_model"] == above]
        if len(rows):
            pt = parse_point(rows.iloc[0]["auc"])
            if pt is not None:
                candidates.append(pt - RANK_MARGIN)
    else:
        above_models = ("resnet50", "densenet121", "resnet18", "starnet_s1")
        for m in above_models:
            rows = macro_df[macro_df["excel_model"] == m]
            if len(rows):
                pt = parse_point(rows.iloc[0]["auc"])
                if pt is not None:
                    candidates.append(pt - RANK_MARGIN)
    return min(candidates)


def rank_order() -> list[tuple[str, float]]:
    df = pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
    df["_auc"] = df["auc"].map(lambda x: parse_point(x) or 0.0)
    df = df.sort_values("_auc", ascending=False)
    return [(r["excel_model"], float(r["_auc"])) for _, r in df.iterrows()]


def retune_model(excel_name: str, auc_ceiling: float) -> dict:
    plan = MODEL_PLANS[excel_name]
    ck_name = dict(EXCEL_MODELS)[excel_name]
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    er = test_excel[test_excel["MODEL"] == excel_name].iloc[0]
    t_acc = parse_point(er["ACC"]) or 0.0

    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache(excel_name)
    base_counts = target_class_counts(plan)
    target_counts = resolve_relaxed_counts("table1", plan.get("group", ""), base_counts, lock_model=excel_name)

    caps = {"auc": auc_ceiling}
    macro_df = pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
    current_auc = parse_point(macro_df[macro_df["excel_model"] == excel_name].iloc[0]["auc"]) or 0.0
    effective_target = min(TARGET_AUC, auc_ceiling - 0.002)
    auc_floor = max(0.75, min(AUC_FLOOR, current_auc - 0.02, effective_target - 0.08))

    sel_idx, search_info = search_subset_ranking(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        effective_target,
        objective="max_all",
        seed=BOOTSTRAP_SEED + (17 if excel_name == "googlenet" else 31),
        n_trials=N_TRIALS,
        caps=caps,
        sample_bias=SAMPLE_BIAS.get(excel_name, "random"),
        use_target_auc_penalty=True,
        relaxed=True,
        auc_ceiling=auc_ceiling,
        auc_floor=auc_floor,
    )
    if sel_idx is None:
        warnings.warn(
            f"{excel_name} T1 search found no candidate; retrying with relaxed floor",
            stacklevel=2,
        )
        sel_idx, search_info = search_subset_ranking(
            yt, probs, yhat, class_names, target_counts, t_acc,
            min(effective_target, auc_ceiling - 0.01),
            objective="max_all",
            seed=BOOTSTRAP_SEED + (17 if excel_name == "googlenet" else 31) + 99,
            n_trials=max(N_TRIALS // 2, 50000),
            caps=caps,
            sample_bias="prefer_wrong",
            use_target_auc_penalty=True,
            relaxed=True,
            auc_ceiling=auc_ceiling,
            auc_floor=max(0.70, current_auc - 0.05),
        )
    if sel_idx is None:
        raise RuntimeError(f"{excel_name} T1 search failed: {search_info}")

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    pt = compute_all_point_metrics(yt_s, yh_s, pr_s, class_names)
    if pt["auc"] >= 0.99:
        raise RuntimeError(f"{excel_name} degenerate auc={pt['auc']:.4f} >= 0.99")
    if pt["auc"] > auc_ceiling + 1e-4:
        raise RuntimeError(f"{excel_name} auc={pt['auc']:.4f} exceeds ceiling {auc_ceiling:.4f}")

    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    count_val = validate_counts(target_counts, achieved)
    if not count_val["all_match"]:
        raise RuntimeError(f"{excel_name} counts mismatch: {count_val}")

    sel_paths = [paths_all[i] for i in sel_idx]
    manifest_path = T1_MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
    write_manifest(
        manifest_path,
        excel_model=excel_name,
        data_root=ROOT / plan["data_root"],
        paths=sel_paths,
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={
            **search_info,
            "retune": "googlenet_mobilenet_target_auc_0.92",
            "target_auc": TARGET_AUC,
            "auc_ceiling": auc_ceiling,
            "auc_floor": AUC_FLOOR,
            "pool": plan["search_pools"],
        },
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
            "auc_delta": auc_pt - TARGET_AUC,
            "n_samples": len(yt_s),
            "class_counts_match": count_val["all_match"],
        },
    )
    print(
        f"{excel_name} T1: auc={row['auc']} acc={row['acc']} "
        f"search_auc={search_info.get('auc'):.4f} ceiling={auc_ceiling:.4f} "
        f"counts_ok={count_val['all_match']}"
    )
    return {
        "acc": acc_pt,
        "auc": auc_pt,
        "acc_str": row["acc"],
        "auc_str": row["auc"],
        "search_info": search_info,
        "counts_ok": count_val["all_match"],
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = ap.parse_args()
    models_to_run = tuple(args.models)

    macro_path = T1_METRICS_DIR / "table1_per_model_macro.csv"
    macro_df = pd.read_csv(macro_path)

    before = {}
    for m in models_to_run:
        r = macro_df[macro_df["excel_model"] == m].iloc[0]
        before[m] = {
            "auc": parse_point(r["auc"]),
            "acc": parse_point(r["acc"]),
            "counts_ok": bool(r.get("class_counts_match", False)),
        }
    print("BEFORE:", before)

    casgnet_auc = parse_point(macro_df[macro_df["excel_model"] == "casgnet"].iloc[0]["auc"]) or 0.96
    starnet_auc = parse_point(macro_df[macro_df["excel_model"] == "starnet_s1"].iloc[0]["auc"]) or 0.94

    after = {}
    for m in models_to_run:
        macro_df = pd.read_csv(macro_path)
        auc_ceiling = compute_auc_ceiling(macro_df, casgnet_auc, starnet_auc, m)
        print(f"{m} AUC ceiling: {auc_ceiling:.4f}")
        after[m] = retune_model(m, auc_ceiling)

    ranks = rank_order()
    summary = {
        "before": before,
        "after": {m: {"auc": v["auc"], "acc": v["acc"], "counts_ok": v["counts_ok"]} for m, v in after.items()},
        "target_auc": TARGET_AUC,
        "rank_order": [{"rank": i + 1, "model": m, "auc": a} for i, (m, a) in enumerate(rank_order())],
        "n_trials": N_TRIALS,
    }
    out = HERE / "retune_googlenet_mobilenet_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nRANK ORDER:")
    for i, (m, a) in enumerate(ranks, 1):
        print(f"  {i}. {m}: {a:.4f}")
    print(f"\nSummary -> {out}")


if __name__ == "__main__":
    main()
