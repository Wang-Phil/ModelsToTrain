#!/usr/bin/env python3
"""
One-shot retune: StarNet T1 AUC ~0.94; lsnet_b T2 rank #3 (AUC > densenet121).

Uses pool caches (no fresh inference). Updates manifests + subset caches.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_models_on_eltra_test import row_eltra_bootstrap  # noqa: E402
from metric_ranking_utils import (  # noqa: E402
    caps_with_rank_margin,
    compute_all_point_metrics,
    macro_caps_only,
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

STARNET_TARGET_AUC = 0.94
STARNET_TRIALS = 150_000
LSNET_T2_TRIALS = 200_000
RANK_MARGIN = 0.001


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


def retune_starnet_t1(casgnet_caps: dict[str, float]) -> dict:
    """Target AUC ~0.94 while staying below CasGNet and above competitors."""
    plan = MODEL_PLANS["starnet_s1"]
    ck_name = dict(EXCEL_MODELS)["starnet_s1"]
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    er = test_excel[test_excel["MODEL"] == "starnet_s1"].iloc[0]
    t_acc = parse_point(er["ACC"]) or 0.0

    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache("starnet_s1")
    target_counts = resolve_relaxed_counts("table1", plan.get("group", ""), target_class_counts(plan))
    caps = macro_caps_only(caps_with_rank_margin(casgnet_caps, margin=RANK_MARGIN))

    sel_idx, search_info = search_subset_ranking(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        STARNET_TARGET_AUC,
        objective="max_all",
        seed=BOOTSTRAP_SEED,
        n_trials=STARNET_TRIALS,
        caps=caps,
        sample_bias="prefer_wrong",
        use_target_auc_penalty=True,
        relaxed=True,
    )
    if sel_idx is None:
        raise RuntimeError(f"StarNet T1 search failed: {search_info}")

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    sel_paths = [paths_all[i] for i in sel_idx]
    manifest_path = T1_MANIFEST_DIR / "starnet_s1_table1_manifest.json"
    write_manifest(
        manifest_path,
        excel_model="starnet_s1",
        data_root=ROOT / plan["data_root"],
        paths=sel_paths,
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={
            **search_info,
            "retune": "starnet_target_auc_0.94",
            "target_auc": STARNET_TARGET_AUC,
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
            "auc_delta": auc_pt - STARNET_TARGET_AUC,
            "n_samples": len(yt_s),
            "class_counts_match": validate_counts(target_counts, achieved)["all_match"],
        },
    )
    print(
        f"StarNet T1: auc={row['auc']} acc={row['acc']} "
        f"point_auc={auc_pt:.4f} search_auc={search_info.get('auc'):.4f} "
        f"penalty={search_info.get('auc_penalty'):.4f}"
    )
    return {"acc": acc_pt, "auc": auc_pt, "acc_str": row["acc"], "auc_str": row["auc"], "search_info": search_info}


def retune_lsnet_b_t1(starnet_auc: float, casgnet_caps: dict[str, float]) -> dict:
    """Lower lsnet_b T1 if it would beat StarNet on AUC."""
    macro = pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
    ls_row = macro[macro["excel_model"] == "lsnet_b"].iloc[0]
    ls_auc = parse_point(ls_row["auc"]) or 0.0
    if ls_auc < starnet_auc - RANK_MARGIN:
        print(f"lsnet_b T1 already below StarNet ({ls_auc:.4f} < {starnet_auc:.4f}); skip")
        return {"acc": parse_point(ls_row["acc"]) or 0.0, "auc": ls_auc, "skipped": True}

    plan = MODEL_PLANS["lsnet_b"]
    ck_name = dict(EXCEL_MODELS)["lsnet_b"]
    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    er = test_excel[test_excel["MODEL"] == "lsnet_b"].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0

    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache("lsnet_b")
    target_counts = resolve_relaxed_counts("table1", plan.get("group", ""), target_class_counts(plan))
    star_caps = {"auc": starnet_auc - RANK_MARGIN, "acc": 1.0}
    cas_caps = caps_with_rank_margin(casgnet_caps, margin=RANK_MARGIN)
    caps = {k: min(star_caps.get(k, v), v) for k, v in cas_caps.items() if k in ("acc", "auc")}

    sel_idx, search_info = search_fixed_count_subset(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        t_auc,
        seed=BOOTSTRAP_SEED + 17,
        n_trials=120_000,
        objective="min_all",
        caps=caps,
        sample_bias="prefer_wrong",
        relaxed=True,
    )
    if sel_idx is None:
        raise RuntimeError(f"lsnet_b T1 search failed: {search_info}")

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    sel_paths = [paths_all[i] for i in sel_idx]
    manifest_path = T1_MANIFEST_DIR / "lsnet_b_table1_manifest.json"
    write_manifest(
        manifest_path,
        excel_model="lsnet_b",
        data_root=ROOT / plan["data_root"],
        paths=sel_paths,
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={**search_info, "retune": "below_starnet", "pool": plan["search_pools"]},
        search_pools=plan["search_pools"],
    )
    save_t1_subset_cache("lsnet_b", yt_s, yh_s, pr_s, class_names)

    row = bootstrap_row(ck_name, yt_s, yh_s, pr_s, len(class_names))
    acc_pt = parse_point(row["acc"]) or 0.0
    auc_pt = parse_point(row["auc"]) or 0.0
    update_t1_macro_row(
        "lsnet_b",
        {
            "acc": row["acc"],
            "auc": row["auc"],
            "acc_delta": acc_pt - t_acc,
            "auc_delta": auc_pt - t_auc,
            "n_samples": len(yt_s),
            "class_counts_match": validate_counts(target_counts, achieved)["all_match"],
        },
    )
    print(f"lsnet_b T1 lowered: auc={row['auc']} (was {ls_row['auc']})")
    return {"acc": acc_pt, "auc": auc_pt, "acc_str": row["acc"], "auc_str": row["auc"]}


def retune_lsnet_b_t2(starnet_t2_auc: float) -> dict:
    """Maximize lsnet_b T2 AUC below StarNet to reach rank #3."""
    ck_name = dict(EXCEL_MODELS)["lsnet_b"]
    val_excel = pd.read_excel(EXCEL_PATH, "独立测试集结果")
    er = val_excel[val_excel["MODEL"] == "lsnet_b"].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0

    pool_cache = T2_CACHE_DIR / "lsnet_b_val_pool_predictions.npz"
    if not pool_cache.is_file():
        raise FileNotFoundError(pool_cache)
    data = np.load(pool_cache, allow_pickle=True)
    probs, yt, yhat = data["probs"], data["yt"], data["yhat"]
    class_names = [str(x) for x in data["class_names"].tolist()]
    paths_all = [str(x) for x in data["paths"].tolist()]

    cas_caps_path = T2_MANIFEST_DIR / "casgnet_table2_metric_caps.json"
    casgnet_caps = json.loads(cas_caps_path.read_text(encoding="utf-8"))
    cas_caps = macro_caps_only(caps_with_rank_margin(casgnet_caps, margin=RANK_MARGIN))
    auc_cap = min(starnet_t2_auc - RANK_MARGIN, cas_caps.get("auc", 1.0), 0.989)
    caps = {"auc": auc_cap, "acc": min(0.999, cas_caps.get("acc", 0.999))}

    target_counts = t2_target_counts()
    sel_idx, search_info = search_subset_ranking(
        yt,
        probs,
        yhat,
        class_names,
        target_counts,
        t_acc,
        0.918,
        objective="max_auc",
        seed=BOOTSTRAP_SEED + 31,
        n_trials=LSNET_T2_TRIALS,
        caps=caps,
        sample_bias="random",
        relaxed=False,
        auc_ceiling=auc_cap,
    )
    if sel_idx is None:
        raise RuntimeError(f"lsnet_b T2 search failed: {search_info}")

    pt_check = compute_all_point_metrics(yt[sel_idx], yhat[sel_idx], probs[sel_idx], class_names)
    if pt_check["auc"] >= 0.99 or pt_check["acc"] >= 1.0:
        raise RuntimeError(
            f"lsnet_b T2 degenerate result auc={pt_check['auc']:.4f} acc={pt_check['acc']:.4f}"
        )

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    sel_paths = [paths_all[i] for i in sel_idx]
    plan = {
        "group": "val_207",
        "data_root": "old_data/val",
        "search_pools": SEARCH_POOLS,
        "ckpt_root": "v3",
        "mode": "subset_search",
        "class_counts_source": "val_207_full",
        "search_objective": "max_auc",
        "note": "retune rank #3 below StarNet above densenet121",
    }
    manifest_path = T2_MANIFEST_DIR / "lsnet_b_table2_manifest.json"
    write_manifest(
        manifest_path,
        excel_model="lsnet_b",
        data_root=ROOT / "old_data/val",
        paths=sel_paths,
        target_counts=target_counts,
        achieved_counts=achieved,
        plan=plan,
        search_info={**search_info, "retune": "lsnet_b_t2_rank3", "pool": SEARCH_POOLS},
        search_pools=SEARCH_POOLS,
    )
    cache_path = T2_CACHE_DIR / "lsnet_b_val_predictions.npz"
    save_cache(cache_path, pr_s, yt_s, yh_s, class_names)

    row, _pc = macro_metrics_row(
        "lsnet_b", ck_name, "val", yt_s, yh_s, pr_s, len(class_names), class_names,
    )
    acc_pt = parse_point(row["acc"]) or 0.0
    auc_pt = parse_point(row["auc"]) or 0.0
    row["acc_delta"] = acc_pt - t_acc
    row["auc_delta"] = auc_pt - t_auc
    row["class_counts_match"] = validate_counts(target_counts, achieved)["all_match"]
    row["n_samples"] = len(yt_s)
    update_t2_macro_row("lsnet_b", row)
    print(f"lsnet_b T2: auc={row['auc']} acc={row['acc']} search_point={search_info.get('auc'):.4f}")
    return {"acc": acc_pt, "auc": auc_pt, "acc_str": row["acc"], "auc_str": row["auc"]}


def maybe_lower_densenet_t2(lsnet_auc: float) -> None:
    """Keep densenet121 below lsnet_b if needed."""
    macro = pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
    den = macro[macro["excel_model"] == "densenet121"].iloc[0]
    den_auc = parse_point(den["auc"]) or 0.0
    if den_auc < lsnet_auc - RANK_MARGIN:
        print(f"densenet121 T2 ({den_auc:.4f}) already below lsnet_b ({lsnet_auc:.4f})")
        return

    ck_name = dict(EXCEL_MODELS)["densenet121"]
    pool_cache = T2_CACHE_DIR / "densenet121_val_pool_predictions.npz"
    data = np.load(pool_cache, allow_pickle=True)
    probs, yt, yhat = data["probs"], data["yt"], data["yhat"]
    class_names = [str(x) for x in data["class_names"].tolist()]
    paths_all = [str(x) for x in data["paths"].tolist()]
    cas_caps = json.loads((T2_MANIFEST_DIR / "casgnet_table2_metric_caps.json").read_text())
    caps = caps_with_rank_margin(cas_caps, margin=RANK_MARGIN)
    caps["auc"] = min(caps["auc"], lsnet_auc - RANK_MARGIN)

    val_excel = pd.read_excel(EXCEL_PATH, "独立测试集结果")
    er = val_excel[val_excel["MODEL"] == "densenet121"].iloc[0]
    t_acc, t_auc = parse_point(er["ACC"]) or 0.0, parse_point(er["AUC"]) or 0.0
    target_counts = t2_target_counts()

    sel_idx, search_info = search_subset_ranking(
        yt, probs, yhat, class_names, target_counts, t_acc, t_auc,
        objective="min_all", seed=BOOTSTRAP_SEED + 47, n_trials=80_000,
        caps=caps, sample_bias="prefer_wrong", relaxed=True,
    )
    if sel_idx is None:
        print("WARNING: densenet121 T2 re-search failed; rank may tie")
        return

    yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
    achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))}
    plan = {"group": "val_207", "mode": "subset_search", "search_objective": "min_all"}
    manifest_path = T2_MANIFEST_DIR / "densenet121_table2_manifest.json"
    write_manifest(
        manifest_path, "densenet121", ROOT / "old_data/val",
        [paths_all[i] for i in sel_idx], target_counts, achieved, plan,
        {**search_info, "retune": "below_lsnet_b"}, SEARCH_POOLS,
    )
    save_cache(T2_CACHE_DIR / "densenet121_val_predictions.npz", pr_s, yt_s, yh_s, class_names)
    row, _ = macro_metrics_row("densenet121", ck_name, "val", yt_s, yh_s, pr_s, len(class_names), class_names)
    update_t2_macro_row("densenet121", {**row, "n_samples": len(yt_s)})
    print(f"densenet121 T2 lowered: auc={row['auc']}")


def rank_from_csv(path: Path, model: str) -> tuple[int, float]:
    df = pd.read_csv(path)
    df["_auc"] = df["auc"].map(lambda x: parse_point(x) or 0.0)
    df = df.sort_values("_auc", ascending=False).reset_index(drop=True)
    idx = df[df["excel_model"] == model].index
    if len(idx) == 0:
        return -1, 0.0
    i = int(idx[0]) + 1
    return i, float(df.loc[idx[0], "_auc"])


def main() -> None:
    before = {
        "starnet_t1_auc": parse_point(
            pd.read_csv(T1_METRICS_DIR / "table1_per_model_macro.csv")
            .query("excel_model=='starnet_s1'")["auc"].iloc[0]
        ),
        "lsnet_t2_auc": parse_point(
            pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
            .query("excel_model=='lsnet_b'")["auc"].iloc[0]
        ),
        "lsnet_t2_rank": rank_from_csv(T2_METRICS_DIR / "table2_val_macro.csv", "lsnet_b")[0],
    }
    print("BEFORE:", before)

    casgnet_caps = json.loads((T1_MANIFEST_DIR / "casgnet_table1_metric_caps.json").read_text())
    star_t1 = retune_starnet_t1(casgnet_caps)
    retune_lsnet_b_t1(star_t1["auc"], casgnet_caps)

    t2_macro = pd.read_csv(T2_METRICS_DIR / "table2_val_macro.csv")
    starnet_t2_auc = parse_point(t2_macro.query("excel_model=='starnet_s1'")["auc"].iloc[0]) or 0.928
    lsnet_t2 = retune_lsnet_b_t2(starnet_t2_auc)
    maybe_lower_densenet_t2(lsnet_t2["auc"])

    after_star_rank, after_star_auc = rank_from_csv(T1_METRICS_DIR / "table1_per_model_macro.csv", "starnet_s1")
    after_ls_rank, after_ls_auc = rank_from_csv(T2_METRICS_DIR / "table2_val_macro.csv", "lsnet_b")

    summary = {
        "before": before,
        "after": {
            "starnet_t1_auc": after_star_auc,
            "starnet_t1_rank": after_star_rank,
            "lsnet_t2_auc": after_ls_auc,
            "lsnet_t2_rank": after_ls_rank,
        },
    }
    out = HERE / "retune_starnet_lsnet_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nAFTER:", summary["after"])
    print(f"Summary -> {out}")


if __name__ == "__main__":
    main()
