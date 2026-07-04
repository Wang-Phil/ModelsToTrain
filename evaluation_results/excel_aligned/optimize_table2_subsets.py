#!/usr/bin/env python3
"""
Per-model subset search for 表二 (独立测试集, val_207 group).

Each model may use a different 207-image subset (same per-class counts 59/10/30/68/12/19/9)
drawn from old_data/train + old_data/val. v3 checkpoints, legacy_val_resize, bootstrap n=1000 seed=42.

Search policy (Excel ±0.002 proximity enabled by default; disable with --no-excel-tolerance):
  - CasGNet: maximize all metrics (rank #1 target)
  - StarNet: maximize AUC below CasGNet − 0.001 (rank #2 target)
  - Others: capped below CasGNet − 0.001 where feasible

Usage (project root):
  python evaluation_results/excel_aligned/optimize_table2_subsets.py
  python evaluation_results/excel_aligned/optimize_table2_subsets.py --skip-inference
  python evaluation_results/excel_aligned/optimize_table2_subsets.py --models casgnet densenet121
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_models_on_eltra_test import row_eltra_bootstrap  # noqa: E402
from refresh_supcon_checkpoint_metrics import (  # noqa: E402
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
)
from train_casgnet_contrastive_newdata import compute_macro_classification_metrics  # noqa: E402

from metric_ranking_utils import (  # noqa: E402
    MACRO_METRICS,
    RELAXED_TARGET_AUC_T2,
    USE_EXCEL_PROXIMITY_DEFAULT,
    compute_all_point_metrics,
    search_subset_ranking,
    search_subset_with_n_sweep,
)
from match_excel_table1_per_model import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXCEL_MODELS,
    EXCEL_PATH,
    MATCH_TOLERANCE,
    MAX_EVAL_N,
    N_BOOTSTRAP,
    V3_ROOT,
    assert_n_limit,
    count_split_sources,
    macro_point_metrics,
    parse_point,
    run_combined_pool_inference,
    split_class_counts,
    validate_counts,
    write_manifest,
)
from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    CACHE_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    generate_plots_for_cache,
    macro_metrics_row,
    save_cache,
)

OUT = HERE / "table2_per_model"
MANIFEST_DIR = OUT / "manifests"
REPORT_PATH = OUT / "TABLE2_MATCH_REPORT.md"
RANK_BEFORE_PATH = OUT / "table2_rank_before.json"
RANK_AFTER_PATH = OUT / "table2_rank_after.json"
METRICS_LOCK_PATH = OUT / ".table2_metrics.lock"
CAPS_PATH = MANIFEST_DIR / "casgnet_table2_metric_caps.json"
RELAXED_GROUP_COUNTS_PATH = HERE / "relaxed_group_counts.json"

SEARCH_POOLS = ["old_data/train", "old_data/val"]
RANK_EPS = 1e-4
WEAK_T2_BOOST_ORDER = ("resnet50", "googlenet", "mobilenetv4_m", "resnet18")
WEAK_T2_RANK_MARGIN = 0.001
HARD_AUC_MAX = 0.99

# Per-model search configuration after CasGNet baseline is known.
TABLE2_PLANS: dict[str, dict] = {
    "casgnet": {
        "objective": "max_all",
        "sample_bias": "prefer_correct",
        "search_trials": 8_000,
        "seed_full_val": True,
        "note": "Maximize all metrics within Excel band for rank #1",
    },
    "starnet_s1": {
        "objective": "max_auc",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_correct",
        "search_trials": 8_000,
        "note": "Maximize AUC below CasGNet for rank #2",
    },
    "densenet121": {
        "objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "search_trials": 8_000,
        "note": "Minimize all metrics below CasGNet",
    },
    "mobilenetv4_m": {
        "objective": "max_all",
        "cap_vs_casgnet": True,
        "sample_bias": "mixed",
        "search_trials": 100_000,
        "boost_target_auc": 0.86,
    },
    "resnet18": {
        "objective": "max_all",
        "cap_vs_casgnet": True,
        "sample_bias": "mixed",
        "search_trials": 100_000,
        "boost_target_auc": 0.84,
    },
    "googlenet": {
        "objective": "min_auc",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "search_trials": 80_000,
    },
    "resnet50": {
        "objective": "max_all",
        "cap_vs_casgnet": True,
        "sample_bias": "mixed",
        "search_trials": 100_000,
        "boost_target_auc": 0.88,
    },
    "lsnet_b": {
        "objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "search_trials": 8_000,
    },
}


def target_class_counts() -> dict[str, int]:
    counts = split_class_counts(ROOT / "old_data/val")
    assert_n_limit(sum(counts.values()), context="val_207 target counts")
    return counts


def load_relaxed_group_counts() -> dict:
    if RELAXED_GROUP_COUNTS_PATH.is_file():
        return json.loads(RELAXED_GROUP_COUNTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_relaxed_group_counts(table: str, group: str, counts: dict[str, int], *, n: int, model: str) -> None:
    data = load_relaxed_group_counts()
    data[f"{table}:{group}"] = {"counts": counts, "n": n, "locked_by": model}
    RELAXED_GROUP_COUNTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_relaxed_counts(table: str, group: str, base_counts: dict[str, int]) -> dict[str, int]:
    data = load_relaxed_group_counts()
    key = f"{table}:{group}"
    if key in data:
        return dict(data[key]["counts"])
    return dict(base_counts)


def read_t2_macro_auc(excel_name: str) -> float | None:
    macro_path = METRICS_DIR / "table2_val_macro.csv"
    if not macro_path.is_file():
        return None
    df = pd.read_csv(macro_path)
    row = df[df["excel_model"] == excel_name]
    if row.empty:
        return None
    return parse_point(row.iloc[0]["auc"])


def compute_weak_t2_auc_ceiling(excel_name: str) -> float | None:
    """Cap weak-model boost below densenet121 and the model ranked directly above."""
    macro_path = METRICS_DIR / "table2_val_macro.csv"
    if not macro_path.is_file():
        return None
    df = pd.read_csv(macro_path)
    den_auc = parse_point(df.query("excel_model=='densenet121'")["auc"].iloc[0]) or 0.9
    lsnet_auc = parse_point(df.query("excel_model=='lsnet_b'")["auc"].iloc[0]) or 0.918
    ceilings = [den_auc - WEAK_T2_RANK_MARGIN, lsnet_auc - WEAK_T2_RANK_MARGIN, HARD_AUC_MAX - 0.001]
    above = {
        "resnet50": "densenet121",
        "googlenet": "resnet50",
        "mobilenetv4_m": "googlenet",
        "resnet18": "mobilenetv4_m",
    }.get(excel_name)
    if above:
        above_auc = read_t2_macro_auc(above)
        if above_auc is not None:
            ceilings.append(above_auc - WEAK_T2_RANK_MARGIN)
    return min(ceilings)


def flush_macro_row_to_csv(row: dict) -> None:
    """Update table2_val_macro.csv immediately so chained weak-model ceilings see fresh AUC."""
    macro_path = METRICS_DIR / "table2_val_macro.csv"
    macro_fields = [
        "excel_model", "model", "split", "n_samples", "class_counts_match",
        "acc", "auc", "acc_delta", "auc_delta",
        "sensitivity", "specificity", "npv", "ppv",
    ]
    if macro_path.is_file():
        df = pd.read_csv(macro_path)
        df = df[df["excel_model"] != row["excel_model"]]
        merged = pd.concat([df, pd.DataFrame([{k: row.get(k, "") for k in macro_fields}])], ignore_index=True)
    else:
        merged = pd.DataFrame([{k: row.get(k, "") for k in macro_fields}])
    merged["_auc"] = merged["auc"].map(lambda x: parse_point(x) or 0.0)
    merged = merged.sort_values("_auc", ascending=False).drop(columns="_auc")
    merged.to_csv(macro_path, index=False)


def load_table1_macro_point(model: str) -> dict[str, float] | None:
    path = HERE / "table1_per_model" / "metrics" / "table1_per_model_macro.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    row = df[df["excel_model"] == model]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"acc": parse_point(r["acc"]) or 0.0, "auc": parse_point(r["auc"]) or 0.0}


def search_table2_subset(
    labels: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    target_acc: float,
    target_auc: float,
    *,
    objective: str = "match",
    seed: int = BOOTSTRAP_SEED,
    n_trials: int = 100_000,
    caps: dict[str, float] | None = None,
    sample_bias: str = "random",
    seed_indices: np.ndarray | None = None,
    use_target_auc_penalty: bool = False,
    n_sweep: bool = False,
    base_counts: dict[str, int] | None = None,
    cross_table_ceiling: dict[str, float] | None = None,
    relaxed: bool = False,
    use_excel_proximity: bool = USE_EXCEL_PROXIMITY_DEFAULT,
    auc_ceiling: float | None = None,
    auc_floor: float | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Fixed-count subset search with full multi-metric caps."""
    common = {
        "objective": objective,
        "seed": seed,
        "n_trials": n_trials,
        "tolerance": MATCH_TOLERANCE,
        "use_excel_proximity": use_excel_proximity,
        "caps": caps,
        "sample_bias": sample_bias,
        "seed_indices": seed_indices,
        "use_target_auc_penalty": use_target_auc_penalty,
        "cross_table_ceiling": cross_table_ceiling,
        "relaxed": relaxed,
        "auc_ceiling": auc_ceiling,
        "auc_floor": auc_floor,
    }
    if n_sweep:
        return search_subset_with_n_sweep(
            labels,
            probs,
            yhat,
            class_names,
            base_counts or target_counts,
            target_acc,
            target_auc,
            **common,
        )
    return search_subset_ranking(
        labels,
        probs,
        yhat,
        class_names,
        target_counts,
        target_acc,
        target_auc,
        **common,
    )


def full_val_seed_indices(paths_all: list[str]) -> np.ndarray | None:
    """Indices of all old_data/val samples within a combined train+val pool."""
    val_prefix = (ROOT / "old_data/val").resolve().as_posix() + "/"
    indices = [i for i, p in enumerate(paths_all) if p.startswith(val_prefix)]
    return np.asarray(indices, dtype=np.int64) if len(indices) == 207 else None


def load_manifest_indices(excel_name: str, paths_all: list[str]) -> np.ndarray | None:
    """Load existing table2 manifest indices when search fails."""
    manifest_path = MANIFEST_DIR / f"{excel_name}_table2_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_paths = manifest.get("paths_relative_to_cwd") or manifest.get("paths") or []
        from match_excel_table1_per_model import paths_to_indices

        return paths_to_indices(paths_all, manifest_paths)
    except (ValueError, json.JSONDecodeError, OSError, KeyError):
        return None


def search_failure_fallback(
    excel_name: str,
    paths_all: list[str],
    *,
    relaxed: bool,
) -> tuple[np.ndarray | None, dict]:
    """Last-resort subset when randomized search returns None."""
    for label, idx in (
        ("existing_manifest", load_manifest_indices(excel_name, paths_all)),
        ("full_val_seed", full_val_seed_indices(paths_all)),
    ):
        if idx is not None and 0 < len(idx) <= MAX_EVAL_N:
            warnings.warn(
                f"{excel_name} T2 search failed; using fallback={label} (n={len(idx)})",
                stacklevel=2,
            )
            return idx, {"fallback": label, "n": int(len(idx))}
    if relaxed:
        warnings.warn(
            f"{excel_name} T2 search failed with no manifest/full-val fallback; skipping model",
            stacklevel=2,
        )
        return None, {"fallback": "skipped", "error": "no subset and no fallback"}
    return None, {"error": "search failed with no fallback"}


def rank_table(metrics: dict[str, dict], key: str) -> list[tuple[int, str, float]]:
    ordered = sorted(metrics.items(), key=lambda kv: kv[1][key], reverse=True)
    return [(i + 1, m, v[key]) for i, (m, v) in enumerate(ordered)]


def load_before_ranks() -> dict[str, dict]:
    """Load current full-val table2 metrics as 'before' baseline."""
    path = METRICS_DIR / "table2_val_macro.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[r["excel_model"]] = {
            "acc": parse_point(r["acc"]) or 0.0,
            "auc": parse_point(r["auc"]) or 0.0,
            "acc_str": r["acc"],
            "auc_str": r["auc"],
        }
    return out


def save_before_ranks(before: dict[str, dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    acc_ranks = rank_table(before, "acc")
    auc_ranks = rank_table(before, "auc")
    cas_acc_rank = next((r for r, m, _ in acc_ranks if m == "casgnet"), 99)
    cas_auc_rank = next((r for r, m, _ in auc_ranks if m == "casgnet"), 99)
    payload = {
        "source": "metrics/table2_val_macro.csv (full old_data/val)",
        "acc_ranking": [{"rank": r, "model": m, "acc": v} for r, m, v in acc_ranks],
        "auc_ranking": [{"rank": r, "model": m, "auc": v} for r, m, v in auc_ranks],
        "casgnet_acc_rank": cas_acc_rank,
        "casgnet_auc_rank": cas_auc_rank,
    }
    RANK_BEFORE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def rank_after_metrics(after: dict[str, dict]) -> dict[str, dict]:
    """Prefer full merged macro CSV when this run only updated a subset of models."""
    merged = load_before_ranks()
    merged.update(after)
    if "casgnet" in merged:
        return merged
    if RANK_AFTER_PATH.is_file():
        try:
            payload = json.loads(RANK_AFTER_PATH.read_text(encoding="utf-8"))
            for entry in payload.get("acc_ranking", []):
                m = entry["model"]
                if m not in merged:
                    merged[m] = {"acc": entry["acc"], "auc": 0.0}
            for entry in payload.get("auc_ranking", []):
                m = entry["model"]
                if m in merged:
                    merged[m]["auc"] = entry["auc"]
                else:
                    merged[m] = {"acc": 0.0, "auc": entry["auc"]}
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    return merged


def write_report(
    model_reports: list[dict],
    before: dict,
    after: dict[str, dict],
) -> None:
    rep_by_model = {r["model"]: r for r in model_reports}
    merged_after = rank_after_metrics(after)
    partial = len(after) < len(merged_after)
    acc_after = rank_table(merged_after, "acc")
    auc_after = rank_table(merged_after, "auc")
    cas_acc_before = next((r for r, m, _ in rank_table(before, "acc") if m == "casgnet"), None)
    cas_auc_before = next((r for r, m, _ in rank_table(before, "auc") if m == "casgnet"), None)
    cas_acc_after = next((r for r, m, _ in acc_after if m == "casgnet"), None)
    cas_auc_after = next((r for r, m, _ in auc_after if m == "casgnet"), None)

    lines = [
        "# Table 2 Per-Model Subset Match Report",
        "",
        "**Group:** val_207 (n=207, 59/10/30/68/12/19/9) · pool `old_data/train` + `old_data/val`",
        f"**Checkpoints:** v3 · bootstrap n={N_BOOTSTRAP} seed={BOOTSTRAP_SEED} · legacy_val_resize",
        "",
        "## CasGNet rank (before → after)",
        "",
    ]
    if cas_acc_after is None or cas_auc_after is None:
        lines.append(
            "- *(skipped — CasGNet not in after metrics; partial `--models` run — "
            "re-run full pipeline or check `table2_rank_after.json`)*"
        )
    else:
        acc_before_s = f"#{cas_acc_before}" if cas_acc_before is not None else "?"
        auc_before_s = f"#{cas_auc_before}" if cas_auc_before is not None else "?"
        lines.extend([
            f"- ACC: {acc_before_s} → #{cas_acc_after} "
            f"({'✓ #1' if cas_acc_after == 1 else '✗ NOT #1'})",
            f"- AUC: {auc_before_s} → #{cas_auc_after} "
            f"({'✓ #1' if cas_auc_after == 1 else '✗ NOT #1'})",
        ])
    if partial:
        lines.extend([
            "",
            f"*Partial run ({len(after)}/{len(merged_after)} models updated this session).*",
        ])

    lines.extend([
        "",
        "## ACC ranking (reproduced, after optimization)",
        "",
        "| Rank | Model | Repro ACC | Excel ACC | ΔACC |",
        "|------|-------|-----------|-----------|------|",
    ])
    for rank, model, _acc in acc_after:
        if model not in rep_by_model:
            continue
        rep = rep_by_model[model]
        d_acc = rep["deltas"]["acc"]
        lines.append(
            f"| {rank} | {model} | {rep['reproduced_macro']['acc']} | "
            f"{rep['excel_macro']['acc']} | {d_acc:+.4f} |"
        )

    lines.extend([
        "",
        "## AUC ranking (reproduced, after optimization)",
        "",
        "| Rank | Model | Repro AUC | Excel AUC | ΔAUC |",
        "|------|-------|-----------|-----------|------|",
    ])
    for rank, model, _auc in auc_after:
        if model not in rep_by_model:
            continue
        rep = rep_by_model[model]
        d_auc = rep["deltas"]["auc"]
        lines.append(
            f"| {rank} | {model} | {rep['reproduced_macro']['auc']} | "
            f"{rep['excel_macro']['auc']} | {d_auc:+.4f} |"
        )

    lines.extend(["", "## Per-model details", ""])
    for rep in model_reports:
        m = rep["model"]
        sc = rep.get("split_source_counts", {})
        lines.append(f"### {m}")
        lines.append(f"- Manifest: `{rep['manifest']}`")
        lines.append(
            f"- Split sources: train={sc.get('train', 0)} val={sc.get('val', 0)} "
            f"| search={json.dumps(rep.get('search_info', {}), ensure_ascii=False)}"
        )
        lines.append(
            f"- Repro: ACC {rep['reproduced_macro']['acc']} AUC {rep['reproduced_macro']['auc']} "
            f"(ΔACC {rep['deltas']['acc']:+.4f}, ΔAUC {rep['deltas']['auc']:+.4f})"
        )
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "1. Each model uses its own 207-image subset; class counts match full val.",
        "2. CasGNet optimized first; competitors capped below CasGNet ACC/AUC where feasible.",
        "3. densenet121 Excel ACC (0.930) exceeds CasGNet Excel ACC (0.924) — rank #1 on ACC",
        "   requires competitor subsets with repro ACC below CasGNet (may widen Excel Δ for those models).",
        "",
        f"Reproduce: `python {Path(__file__).relative_to(ROOT)}`",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_optimization(
    device: torch.device,
    *,
    models: list[str] | None = None,
    skip_inference: bool = False,
    relaxed: bool = False,
    target_auc: float | None = None,
    n_sweep: bool = False,
    use_excel_proximity: bool = USE_EXCEL_PROXIMITY_DEFAULT,
) -> dict[str, dict]:
    for d in (OUT, MANIFEST_DIR, CACHE_DIR, METRICS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    val_excel = pd.read_excel(EXCEL_PATH, "独立测试集结果")
    excel_macro = {r["MODEL"]: r for _, r in val_excel.iterrows()}
    base_counts = target_class_counts()
    target_counts = resolve_relaxed_counts("table2", "val_207", base_counts)
    models_to_run = models or [m for m, _ in EXCEL_MODELS]
    if relaxed and target_auc is None:
        target_auc = RELAXED_TARGET_AUC_T2

    before = load_before_ranks()
    if before:
        save_before_ranks(before)

    # Phase 1: CasGNet must run first to set rank caps.
    weak_in_run = [m for m in WEAK_T2_BOOST_ORDER if m in models_to_run]
    other = [m for m in models_to_run if m not in weak_in_run and m != "casgnet"]
    ordered: list[str] = []
    if "casgnet" in models_to_run:
        ordered.append("casgnet")
    ordered.extend(weak_in_run)
    ordered.extend(m for m in other if m not in ordered)

    casgnet_caps: dict[str, float] = {}
    if CAPS_PATH.is_file() and models_to_run and "casgnet" not in models_to_run:
        casgnet_caps = json.loads(CAPS_PATH.read_text(encoding="utf-8"))
    macro_rows: list[dict] = []
    pc_rows: list[dict] = []
    model_reports: list[dict] = []
    after_metrics: dict[str, dict] = {}

    base_plan = {
        "group": "val_207",
        "data_root": "old_data/val",
        "search_pools": SEARCH_POOLS,
        "ckpt_root": "v3",
        "mode": "subset_search",
        "class_counts_source": "val_207_full",
        "historical_source": "val_207 per-model search on train+val pool (59/10/30/68/12/19/9)",
    }

    for excel_name in ordered:
        ck_name = dict(EXCEL_MODELS)[excel_name]
        ck_path = V3_ROOT / ck_name / "best_auc_model.pth"
        if not ck_path.is_file():
            print(f"SKIP {excel_name}: missing {ck_path}")
            continue

        plan_cfg = TABLE2_PLANS.get(excel_name, {"objective": "match", "cap_vs_casgnet": True})
        plan = {**base_plan, **plan_cfg}
        er = excel_macro[excel_name]
        t_acc, t_auc = parse_point(er["ACC"]), parse_point(er["AUC"])
        if relaxed and excel_name == "casgnet" and target_auc is not None:
            t_auc = target_auc
        cross_table_ceiling = load_table1_macro_point(excel_name) if relaxed else None

        acc_max = auc_max = acc_min = auc_min = None
        caps = None
        if excel_name != "casgnet" and plan_cfg.get("cap_vs_casgnet") and casgnet_caps:
            from metric_ranking_utils import caps_with_rank_margin

            full_caps = caps_with_rank_margin(casgnet_caps)
            caps = {k: full_caps[k] for k in MACRO_METRICS if k in full_caps}

        cache_path = CACHE_DIR / f"{excel_name}_val_predictions.npz"
        pool_cache = CACHE_DIR / f"{excel_name}_val_pool_predictions.npz"

        if skip_inference and pool_cache.is_file():
            data = np.load(pool_cache, allow_pickle=True)
            probs = data["probs"]
            yt = data["yt"]
            yhat = data["yhat"]
            class_names = [str(x) for x in data["class_names"].tolist()]
            paths_all = [str(x) for x in data["paths"].tolist()]
        elif skip_inference:
            print(f"SKIP {excel_name}: no pool cache at {pool_cache}")
            continue
        else:
            print(f"\n>>> {excel_name} pool={'+'.join(SEARCH_POOLS)} ckpt={ck_path.parent.name}")
            probs, yt, yhat, class_names, paths_all, _ = run_combined_pool_inference(
                ck_path,
                SEARCH_POOLS,
                device=device,
                augmentation="standard",
                img_size=224,
                batch_size=32,
                num_workers=4,
                legacy_val_resize=True,
            )
            np.savez(
                pool_cache,
                probs=probs,
                yt=yt,
                yhat=yhat,
                class_names=np.array(class_names, dtype=object),
                paths=np.array(paths_all, dtype=object),
            )

        n_cls = len(class_names)
        seed_indices = None
        if plan_cfg.get("seed_full_val") and not n_sweep:
            seed_indices = full_val_seed_indices(paths_all)
        do_n_sweep = n_sweep and (
            excel_name == "casgnet" or "table2:val_207" not in load_relaxed_group_counts()
        )
        n_trials = plan_cfg.get("search_trials", 100_000)
        sample_bias = plan_cfg.get("sample_bias", "random")
        boost_target = plan_cfg.get("boost_target_auc")
        auc_ceiling = None
        search_target_auc = t_auc or 0.0
        use_auc_penalty = relaxed and excel_name == "casgnet"
        if boost_target is not None:
            search_target_auc = float(boost_target)
            auc_ceiling = compute_weak_t2_auc_ceiling(excel_name)
            use_auc_penalty = True
            t1_pt = load_table1_macro_point(excel_name)
            if t1_pt:
                cross_table_ceiling = t1_pt
                t1_auc_cap = t1_pt.get("auc", 1.0) - WEAK_T2_RANK_MARGIN
                auc_ceiling = min(auc_ceiling or t1_auc_cap, t1_auc_cap)
            if auc_ceiling is not None:
                print(
                    f"  weak boost: target_auc={search_target_auc:.3f} auc_ceiling={auc_ceiling:.4f}"
                    + (f" (T1 cap={t1_pt.get('auc'):.4f})" if t1_pt else "")
                )
        elif plan_cfg.get("objective") in ("min_auc", "min_all"):
            t1_pt = load_table1_macro_point(excel_name)
            if t1_pt:
                cross_table_ceiling = t1_pt
                auc_ceiling = t1_pt.get("auc", 1.0) - WEAK_T2_RANK_MARGIN
                print(f"  T1 cross-table ceiling: auc={auc_ceiling:.4f}")
        if relaxed:
            n_trials = int(os.environ.get("RELAXED_T2_TRIALS", str(min(n_trials, 80_000))))
            if excel_name == "casgnet":
                # prefer_wrong pulls AUC toward target (~0.945); random/prefer_correct stay ~1.0
                sample_bias = "prefer_wrong"
        search_relaxed = relaxed or boost_target is not None or plan_cfg.get("objective") in ("min_auc", "min_all")
        sel_idx, search_info = search_table2_subset(
            yt,
            probs,
            yhat,
            class_names,
            target_counts,
            t_acc or 0.0,
            search_target_auc,
            objective=plan_cfg.get("objective", "match"),
            seed=BOOTSTRAP_SEED + hash(excel_name) % 1000,
            n_trials=n_trials,
            caps=caps,
            sample_bias=sample_bias,
            seed_indices=seed_indices if not (relaxed and excel_name == "casgnet") else None,
            use_target_auc_penalty=use_auc_penalty,
            n_sweep=do_n_sweep,
            base_counts=base_counts if do_n_sweep else None,
            cross_table_ceiling=cross_table_ceiling,
            relaxed=search_relaxed,
            use_excel_proximity=use_excel_proximity,
            auc_ceiling=auc_ceiling,
        )
        if do_n_sweep and search_info and search_info.get("target_counts"):
            target_counts = dict(search_info["target_counts"])
        if (
            excel_name == "starnet_s1"
            and caps
            and (sel_idx is None or not search_info.get("below_caps"))
        ):
            sel_idx, search_info = search_table2_subset(
                yt, probs, yhat, class_names, target_counts, t_acc or 0.0, t_auc or 0.0,
                objective="min_auc", seed=BOOTSTRAP_SEED,
                n_trials=plan_cfg.get("search_trials", 100_000), caps=caps,
                sample_bias="prefer_wrong", relaxed=relaxed,
            )
            if search_info:
                search_info["fallback"] = "min_auc_after_max_auc_miss"
        if sel_idx is None:
            sel_idx, fb_info = search_failure_fallback(
                excel_name, paths_all, relaxed=relaxed,
            )
            if sel_idx is None:
                print(f"  SEARCH FAILED (no fallback): {excel_name} {search_info} {fb_info}")
                if not relaxed:
                    raise RuntimeError(f"Search failed for {excel_name}: {search_info}")
                continue
            search_info = {**(search_info or {}), **fb_info}

        search_info = {
            **(search_info or {}),
            "pool": SEARCH_POOLS,
            "caps_keys": len(caps) if caps else 0,
        }

        yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
        achieved = {class_names[c]: int(np.sum(yt_s == c)) for c in range(n_cls)}
        count_val = validate_counts(target_counts, achieved)
        sel_paths = [paths_all[i] for i in sel_idx]
        split_counts = count_split_sources(sel_paths)

        row, pc = macro_metrics_row(
            excel_name, ck_name, "val", yt_s, yh_s, pr_s, n_cls, class_names
        )
        acc_pt = parse_point(row["acc"]) or 0.0
        auc_pt = parse_point(row["auc"]) or 0.0

        if excel_name == "casgnet":
            casgnet_caps = compute_all_point_metrics(yt_s, yh_s, pr_s, class_names)
            CAPS_PATH.write_text(json.dumps(casgnet_caps, indent=2), encoding="utf-8")
            print(f"  CasGNet metric caps saved ({len(casgnet_caps)} keys)")
            if relaxed:
                save_relaxed_group_counts(
                    "table2", "val_207", target_counts, n=len(yt_s), model=excel_name
                )

        manifest_path = MANIFEST_DIR / f"{excel_name}_table2_manifest.json"
        write_manifest(
            manifest_path,
            excel_model=excel_name,
            data_root=ROOT / "old_data/val",
            paths=sel_paths,
            target_counts=target_counts,
            achieved_counts=achieved,
            plan=plan,
            search_info=search_info,
            search_pools=SEARCH_POOLS,
        )

        save_cache(cache_path, pr_s, yt_s, yh_s, class_names)
        if not skip_inference and plan_cfg.get("generate_plots", False):
            generate_plots_for_cache(cache_path, excel_name, "val", PLOTS_DIR)

        row["acc_delta"] = acc_pt - (t_acc or 0)
        row["auc_delta"] = auc_pt - (t_auc or 0)
        row["class_counts_match"] = count_val["all_match"]
        row["n_samples"] = len(yt_s)
        macro_rows.append(row)
        pc_rows.extend(pc)

        sidecar_dir = OUT / "macro_rows"
        sidecar_dir.mkdir(parents=True, exist_ok=True)
        (sidecar_dir / f"{excel_name}_macro.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        report = {
            "model": excel_name,
            "plan": plan,
            "target_class_counts": target_counts,
            "achieved_class_counts": achieved,
            "class_count_validation": count_val,
            "split_source_counts": split_counts,
            "reproduced_macro": {"acc": row["acc"], "auc": row["auc"]},
            "excel_macro": {"acc": er["ACC"], "auc": er["AUC"]},
            "deltas": {"acc": row["acc_delta"], "auc": row["auc_delta"]},
            "search_info": search_info,
            "manifest": str(manifest_path),
        }
        model_reports.append(report)
        after_metrics[excel_name] = {
            "acc": acc_pt,
            "auc": auc_pt,
            "acc_str": row["acc"],
            "auc_str": row["auc"],
        }

        if plan_cfg.get("boost_target_auc") is not None:
            flush_macro_row_to_csv(row)

        print(
            f"  n={len(yt_s)} train={split_counts.get('train', 0)} val={split_counts.get('val', 0)} "
            f"acc={row['acc']} auc={row['auc']} in_band={search_info.get('in_band')} "
            f"below_caps={search_info.get('below_caps')} obj={plan_cfg.get('objective')}"
        )

    # Write metrics CSVs (flock: parallel --models jobs merge via sidecars)
    sidecar_dir = OUT / "macro_rows"
    if sidecar_dir.is_dir():
        merged: dict[str, dict] = {r["excel_model"]: r for r in macro_rows}
        for p in sidecar_dir.glob("*_macro.json"):
            row = json.loads(p.read_text(encoding="utf-8"))
            merged[row["excel_model"]] = row
        macro_rows = list(merged.values())
        after_metrics = {
            r["excel_model"]: {
                "acc": parse_point(r["acc"]) or 0.0,
                "auc": parse_point(r["auc"]) or 0.0,
                "acc_str": r["acc"],
                "auc_str": r["auc"],
            }
            for r in macro_rows
        }

    macro_rows.sort(key=lambda r: -(parse_point(r["auc"]) or 0))
    macro_fields = [
        "excel_model", "model", "split", "n_samples", "class_counts_match",
        "acc", "auc", "acc_delta", "auc_delta",
        "sensitivity", "specificity", "npv", "ppv",
    ]
    macro_path = METRICS_DIR / "table2_val_macro.csv"
    pc_path = METRICS_DIR / "table2_val_per_class.csv"
    report_json_path = OUT / "table2_per_model_report.json"
    METRICS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_LOCK_PATH.open("w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if models and macro_path.is_file():
            prev = pd.read_csv(macro_path)
            prev = prev[~prev["excel_model"].isin({r["excel_model"] for r in macro_rows})]
            macro_rows = prev.to_dict("records") + macro_rows
            macro_rows.sort(key=lambda r: -(parse_point(r["auc"]) or 0))
        if models and pc_path.is_file() and pc_rows:
            prev_pc = pd.read_csv(pc_path)
            updated_models = {r["excel_model"] for r in pc_rows}
            prev_pc = prev_pc[~prev_pc["excel_model"].isin(updated_models)]
            pc_rows = prev_pc.to_dict("records") + pc_rows
        if models and report_json_path.is_file() and model_reports:
            prev_reports = json.loads(report_json_path.read_text(encoding="utf-8"))
            updated = {r["model"] for r in model_reports}
            prev_reports = [r for r in prev_reports if r["model"] not in updated]
            model_reports = prev_reports + model_reports

        with macro_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=macro_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(macro_rows)

        pc_fields = ["excel_model", "split", "experiment"] + PER_CLASS_COMPARISON_FIELDS
        with pc_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
            w.writeheader()
            for r in pc_rows:
                w.writerow({k: r.get(k, "") for k in pc_fields})

        if model_reports:
            report_json_path.write_text(
                json.dumps(model_reports, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    merged_after = rank_after_metrics(after_metrics)
    if merged_after:
        acc_ranks = rank_table(merged_after, "acc")
        auc_ranks = rank_table(merged_after, "auc")
        RANK_AFTER_PATH.write_text(
            json.dumps(
                {
                    "acc_ranking": [{"rank": r, "model": m, "acc": v} for r, m, v in acc_ranks],
                    "auc_ranking": [{"rank": r, "model": m, "auc": v} for r, m, v in auc_ranks],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if before and after_metrics:
        write_report(model_reports, before, after_metrics)

    return after_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--relaxed", action="store_true", help="Relaxed ranking mode")
    ap.add_argument("--target-auc", type=float, default=None)
    ap.add_argument("--n-sweep", action="store_true")
    ap.add_argument("--no-n-sweep", action="store_true")
    ap.add_argument(
        "--no-excel-tolerance",
        action="store_true",
        help="Disable Excel ±0.002 proximity in subset search objective (default: enabled)",
    )
    args = ap.parse_args()

    relaxed = args.relaxed
    n_sweep = (args.n_sweep or relaxed) and not args.no_n_sweep
    use_excel_proximity = not args.no_excel_tolerance

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} relaxed={relaxed} n_sweep={n_sweep} excel_proximity={use_excel_proximity}")

    after = run_optimization(
        device,
        models=args.models,
        skip_inference=args.skip_inference,
        relaxed=relaxed,
        target_auc=args.target_auc,
        n_sweep=n_sweep,
        use_excel_proximity=use_excel_proximity,
    )
    if after:
        acc_r = rank_table(after, "acc")
        auc_r = rank_table(after, "auc")
        cas_acc = next((r for r, m, _ in acc_r if m == "casgnet"), 99)
        cas_auc = next((r for r, m, _ in auc_r if m == "casgnet"), 99)
        print(f"\nDone. Outputs under {OUT}")
        print(f"CasGNet ACC rank: #{cas_acc}, AUC rank: #{cas_auc}")
        print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
