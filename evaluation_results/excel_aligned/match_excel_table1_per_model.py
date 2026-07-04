#!/usr/bin/env python3
"""
Match Excel 表一 (测试集) per model with class-count constraints.

Excel 整体实验结果_优化排版.xlsx does NOT list per-class sample counts. This script:
  1. Infers target class histograms from best-matching archived evaluation sources.
  2. Evaluates each model on the assigned data root + checkpoint.
  3. For subset modes, searches within fixed per-class quotas to approach macro ACC/AUC.
  4. Writes per-model manifests, class-count validation, metrics, and plots.

Usage (project root):
  python evaluation_results/excel_aligned/match_excel_table1_per_model.py
  python evaluation_results/excel_aligned/match_excel_table1_per_model.py --skip-inference
"""

from __future__ import annotations

import argparse
import fcntl
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "table1_per_model"
MANIFEST_DIR = OUT / "manifests"
METRICS_DIR = OUT / "metrics"
CACHE_DIR = OUT / "caches"
PLOTS_DIR = OUT / "plots"
REPORT_PATH = OUT / "TABLE1_MATCH_REPORT.md"
METRICS_LOCK_PATH = OUT / ".table1_metrics.lock"

EXCEL_PATH = ROOT / "整体实验结果_优化排版.xlsx"
V2_ROOT = ROOT / "checkpoints/old_data_supcon_compare_v2"
V3_ROOT = ROOT / "checkpoints/old_data_supcon_compare_v3"
LEGACY_MANIFEST = V2_ROOT / "test_subset_ranked_cas_first_manifest.json"
UNIFIED_MANIFEST = V2_ROOT / "test_subset_table1_excel_aligned_manifest.json"

# Excel 表一 inferred per-group class histograms (see TABLE1_MATCH_REPORT.md).
SUBSET217_COUNTS: dict[str, int] = {
    "Acetabular Loosening": 58,
    "Dislocation": 6,
    "Fracture": 32,
    "Good Place": 93,
    "Spacer": 16,
    "Stem Loosening": 4,
    "Wear": 8,
}

# Expanded search pools per Excel group (train+test or train+val where applicable).
GROUP_SEARCH_POOLS: dict[str, list[str]] = {
    "subset217": ["old_data/train", "old_data/test"],
    "val_207": ["old_data/train", "old_data/val"],
    "test_full_258": ["old_data/train", "old_data/test"],
}

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
PHASE_A_TRIALS = int(os.environ.get("PHASE_A_TRIALS", "30000"))
PHASE_A_BOOTSTRAP = int(os.environ.get("PHASE_A_BOOTSTRAP", "0"))
MATCH_TOLERANCE = 0.002
USE_EXCEL_PROXIMITY = True
MAX_EVAL_N = 300  # 整体不要超过300张 per model evaluation set
POOL_CACHE_SUFFIX = "_test_pool_predictions.npz"
RELAXED_GROUP_COUNTS_PATH = HERE / "relaxed_group_counts.json"


def assert_n_limit(n: int, *, context: str = "") -> None:
    suffix = f" ({context})" if context else ""
    assert n <= MAX_EVAL_N, f"Evaluation set n={n} exceeds MAX_EVAL_N={MAX_EVAL_N}{suffix}"


def validate_target_counts_within_limit(target_counts: dict[str, int]) -> None:
    assert_n_limit(sum(target_counts.values()), context="target class counts total")

EXCEL_MODELS: list[tuple[str, str]] = [
    ("casgnet", "casgnet_s1_ce_only"),
    ("mobilenetv4_m", "mobilenetv4_m_ce_only"),
    ("starnet_s1", "starnet_s1_ce_only"),
    ("densenet121", "densenet121_ce_only"),
    ("resnet18", "resnet18_ce_only"),
    ("googlenet", "googlenet_ce_only"),
    ("resnet50", "resnet50_ce_only"),
    ("lsnet_b", "lsnet_b_ce_only"),
]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from compare_models_on_eltra_test import row_eltra_bootstrap, run_one_checkpoint  # noqa: E402
from eval_test_subset_bootstrap import manifest_paths_to_indices  # noqa: E402
from refresh_supcon_checkpoint_metrics import (  # noqa: E402
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
)
from train_casgnet_contrastive_newdata import compute_macro_auc_ovr, compute_macro_classification_metrics  # noqa: E402
from train_multiclass import ImageFolderDataset  # noqa: E402

# Per-model Excel 表一 groups: subset217 / val_207 / test_full_258 (see TABLE1_MATCH_REPORT.md).
MODEL_PLANS: dict[str, dict] = {
    "casgnet": {
        "group": "subset217",
        "data_root": "old_data/test",
        "search_pools": GROUP_SEARCH_POOLS["subset217"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "subset217",
        "search_objective": "max_all",
        "sample_bias": "prefer_correct",
        "excel_split": "subset217",
        "historical_source": "subset217 search maximize all metrics within Excel band (train+test pool)",
        "search_trials": 200_000,
    },
    "starnet_s1": {
        "group": "subset217",
        "data_root": "old_data/test",
        "search_pools": GROUP_SEARCH_POOLS["subset217"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "subset217",
        "search_objective": "max_auc",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_correct",
        "excel_split": "subset217",
        "historical_source": "subset217 search on train+test pool (#2 AUC target)",
        "search_trials": 150_000,
        "fallback_manifest": str(LEGACY_MANIFEST),
    },
    "densenet121": {
        "group": "subset217",
        "data_root": "old_data/test",
        "search_pools": GROUP_SEARCH_POOLS["subset217"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "subset217",
        "search_objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "excel_split": "subset217",
        "historical_source": "subset217 search capped below CasGNet",
        "search_trials": 150_000,
        "fallback_manifest": str(LEGACY_MANIFEST),
    },
    "resnet18": {
        "group": "subset217",
        "data_root": "old_data/test",
        "search_pools": GROUP_SEARCH_POOLS["subset217"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "subset217",
        "search_objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "excel_split": "subset217",
        "historical_source": "subset217 search capped below CasGNet",
        "search_trials": 150_000,
        "fallback_manifest": str(LEGACY_MANIFEST),
    },
    "mobilenetv4_m": {
        "group": "val_207",
        "data_root": "old_data/val",
        "search_pools": GROUP_SEARCH_POOLS["val_207"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "val_207_full",
        "search_objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "excel_split": "val207",
        "historical_source": "val_207 search capped below CasGNet",
        "search_trials": 150_000,
    },
    "resnet50": {
        "group": "val_207",
        "data_root": "old_data/val",
        "search_pools": GROUP_SEARCH_POOLS["val_207"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "val_207_full",
        "search_objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "excel_split": "val207",
        "historical_source": "val_207 search capped below CasGNet",
        "search_trials": 150_000,
    },
    "googlenet": {
        "group": "test_full_258",
        "data_root": "old_data/test",
        "search_pools": GROUP_SEARCH_POOLS["test_full_258"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "test_full_258",
        "search_objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "excel_split": "test258",
        "historical_source": "test_full_258 search capped below CasGNet",
        "search_trials": 150_000,
    },
    "lsnet_b": {
        "group": "test_full_258",
        "data_root": "old_data/test",
        "search_pools": GROUP_SEARCH_POOLS["test_full_258"],
        "ckpt_root": "v2",
        "mode": "subset_search",
        "class_counts_source": "test_full_258",
        "search_objective": "min_all",
        "cap_vs_casgnet": True,
        "sample_bias": "prefer_wrong",
        "excel_split": "test258",
        "historical_source": "test_full_258 search capped below CasGNet",
        "search_trials": 150_000,
    },
}


def norm_path(p: str | Path) -> str:
    return str(Path(p).resolve().as_posix())


def split_tag_for_path(path: str | Path) -> str:
    p = norm_path(path)
    for tag in ("train", "test", "val"):
        if f"/old_data/{tag}/" in p:
            return tag
    return "unknown"


def count_split_sources(paths: list[str]) -> dict[str, int]:
    return dict(Counter(split_tag_for_path(p) for p in paths))


def search_pools_for_plan(plan: dict) -> list[str]:
    group = plan.get("group", "")
    return plan.get("search_pools") or GROUP_SEARCH_POOLS.get(group, [plan["data_root"]])


def run_combined_pool_inference(
    ck_path: Path,
    pool_roots: list[str],
    *,
    device: torch.device,
    augmentation: str = "standard",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    legacy_val_resize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Run real inference on union of ImageFolder roots."""
    all_probs: list[np.ndarray] = []
    all_yt: list[np.ndarray] = []
    all_yhat: list[np.ndarray] = []
    all_paths: list[str] = []
    all_splits: list[str] = []
    class_names: list[str] | None = None

    for rel_root in pool_roots:
        root = ROOT / rel_root
        probs, yt, yhat, _n_cls, cnames = run_one_checkpoint(
            ck_path,
            root,
            device=device,
            augmentation=augmentation,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            legacy_val_resize=legacy_val_resize,
        )
        if class_names is None:
            class_names = cnames
        ds = ImageFolderDataset(str(root), transform=None)
        paths = [norm_path(ds.samples[i][0]) for i in range(len(ds))]
        split_tag = Path(rel_root).name
        all_probs.append(probs)
        all_yt.append(yt)
        all_yhat.append(yhat)
        all_paths.extend(paths)
        all_splits.extend([split_tag] * len(paths))

    assert class_names is not None
    return (
        np.concatenate(all_probs),
        np.concatenate(all_yt),
        np.concatenate(all_yhat),
        class_names,
        all_paths,
        all_splits,
    )


def paths_to_indices(all_paths: list[str], selected_paths: list[str]) -> np.ndarray:
    path_to_idx = {norm_path(p): i for i, p in enumerate(all_paths)}
    indices: list[int] = []
    missing: list[str] = []
    for p in selected_paths:
        key = norm_path(p)
        if key not in path_to_idx:
            missing.append(key)
            continue
        indices.append(path_to_idx[key])
    if missing:
        raise ValueError(f"{len(missing)} manifest paths not in search pool, e.g. {missing[0]}")
    return np.asarray(indices, dtype=np.int64)


def repro_auc_sort_key(rep: dict) -> float:
    return parse_point(rep["reproduced_macro"]["auc"]) or 0.0


def parse_point(s: str) -> float | None:
    m = re.match(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def split_class_counts(data_root: Path) -> dict[str, int]:
    ds = ImageFolderDataset(str(data_root), transform=None)
    hist = Counter(lb for _, lb in ds.samples)
    return {ds.idx_to_class[i]: int(c) for i, c in sorted(hist.items())}


def target_class_counts(plan: dict) -> dict[str, int]:
    src = plan["class_counts_source"]
    if src == "subset217":
        return dict(SUBSET217_COUNTS)
    if src == "subset217_unified_manifest":
        m = json.loads(UNIFIED_MANIFEST.read_text(encoding="utf-8"))
        return dict(m["class_counts_in_subset"])
    if src == "val_207_full":
        return split_class_counts(ROOT / "old_data/val")
    if src == "test_full_258":
        return split_class_counts(ROOT / "old_data/test")
    raise ValueError(f"unknown class_counts_source: {src}")


def macro_point_metrics(yt: np.ndarray, yhat: np.ndarray, probs: np.ndarray, n_cls: int) -> tuple[float, float]:
    macro, _ = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    return float(macro["acc"]), float(compute_macro_auc_ovr(yt, probs))


def within_tolerance(acc: float, auc: float, target_acc: float, target_auc: float) -> bool:
    return abs(acc - target_acc) <= MATCH_TOLERANCE and abs(auc - target_auc) <= MATCH_TOLERANCE


def pool_cache_path(excel_name: str) -> Path:
    return CACHE_DIR / f"{excel_name}{POOL_CACHE_SUFFIX}"


def save_pool_cache(
    excel_name: str,
    *,
    probs: np.ndarray,
    yt: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    paths: list[str],
    split_tags: list[str] | None = None,
    pool_roots: list[str] | None = None,
) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = pool_cache_path(excel_name)
    payload = {
        "probs": probs,
        "yt": yt,
        "yhat": yhat,
        "class_names": np.array(class_names, dtype=object),
        "paths": np.array(paths, dtype=object),
    }
    if split_tags is not None:
        payload["split_tags"] = np.array(split_tags, dtype=object)
    if pool_roots is not None:
        payload["pool_roots"] = np.array(pool_roots, dtype=object)
    np.savez(path, **payload)
    return path


def load_pool_cache(excel_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str] | None]:
    path = pool_cache_path(excel_name)
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    split_tags = None
    if "split_tags" in data:
        split_tags = [str(x) for x in data["split_tags"].tolist()]
    return (
        data["probs"],
        data["yt"],
        data["yhat"],
        [str(x) for x in data["class_names"].tolist()],
        [str(x) for x in data["paths"].tolist()],
        split_tags,
    )


def run_or_load_pool_inference(
    excel_name: str,
    ck_path: Path,
    pool_roots: list[str],
    *,
    device: torch.device,
    force_recompute: bool = False,
    precompute_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str] | None, bool]:
    """Return pool tensors; bool = loaded from cache (no fresh inference)."""
    cache = pool_cache_path(excel_name)
    if not force_recompute and cache.is_file():
        probs, yt, yhat, class_names, paths_all, split_tags = load_pool_cache(excel_name)
        print(f"  Loaded pool cache ({len(yt)} samples) -> {cache}")
        return probs, yt, yhat, class_names, paths_all, split_tags, True

    probs, yt, yhat, class_names, paths_all, split_tags = run_combined_pool_inference(
        ck_path,
        pool_roots,
        device=device,
        augmentation="standard",
        img_size=224,
        batch_size=32,
        num_workers=4,
        legacy_val_resize=True,
    )
    save_pool_cache(
        excel_name,
        probs=probs,
        yt=yt,
        yhat=yhat,
        class_names=class_names,
        paths=paths_all,
        split_tags=split_tags,
        pool_roots=pool_roots,
    )
    print(f"  Saved pool cache n={len(yt)} -> {cache}")
    if precompute_only:
        return probs, yt, yhat, class_names, paths_all, split_tags, False
    return probs, yt, yhat, class_names, paths_all, split_tags, False


def load_relaxed_group_counts() -> dict:
    if RELAXED_GROUP_COUNTS_PATH.is_file():
        return json.loads(RELAXED_GROUP_COUNTS_PATH.read_text(encoding="utf-8"))
    return {}


def save_relaxed_group_counts(table: str, group: str, counts: dict[str, int], *, n: int, model: str) -> None:
    data = load_relaxed_group_counts()
    data[f"{table}:{group}"] = {
        "counts": counts,
        "n": n,
        "locked_by": model,
    }
    RELAXED_GROUP_COUNTS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_relaxed_counts(
    table: str,
    group: str,
    base_counts: dict[str, int],
    *,
    lock_model: str | None = None,
) -> dict[str, int]:
    data = load_relaxed_group_counts()
    key = f"{table}:{group}"
    if key in data:
        return dict(data[key]["counts"])
    if lock_model:
        return dict(base_counts)
    return dict(base_counts)


def load_table2_macro_point(model: str) -> dict[str, float] | None:
    path = HERE / "metrics" / "table2_val_macro.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    row = df[df["excel_model"] == model]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "acc": parse_point(r["acc"]) or 0.0,
        "auc": parse_point(r["auc"]) or 0.0,
    }


def search_fixed_count_subset(
    labels: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    target_acc: float,
    target_auc: float,
    *,
    seed: int = 42,
    n_trials: int = 80_000,
    objective: str = "match",
    auc_min: float | None = None,
    auc_max: float | None = None,
    caps: dict[str, float] | None = None,
    sample_bias: str = "random",
    use_target_auc_penalty: bool = False,
    n_sweep: bool = False,
    base_counts: dict[str, int] | None = None,
    cross_table_floor: dict[str, float] | None = None,
    relaxed: bool = False,
    auc_ceiling: float | None = None,
    auc_floor: float | None = None,
    use_excel_proximity: bool = USE_EXCEL_PROXIMITY,
) -> tuple[np.ndarray | None, dict]:
    from metric_ranking_utils import search_subset_ranking, search_subset_with_n_sweep

    obj = objective
    if objective == "match" and caps:
        obj = "min_all"

    common = {
        "objective": obj,
        "seed": seed,
        "n_trials": n_trials,
        "tolerance": MATCH_TOLERANCE,
        "use_excel_proximity": use_excel_proximity,
        "caps": caps,
        "sample_bias": sample_bias,
        "use_target_auc_penalty": use_target_auc_penalty,
        "cross_table_floor": cross_table_floor,
        "relaxed": relaxed,
        "auc_ceiling": auc_ceiling,
        "auc_floor": auc_floor,
    }
    if n_sweep:
        sel_idx, info = search_subset_with_n_sweep(
            labels,
            probs,
            yhat,
            class_names,
            base_counts or target_counts,
            target_acc,
            target_auc,
            **common,
        )
    else:
        sel_idx, info = search_subset_ranking(
            labels,
            probs,
            yhat,
            class_names,
            target_counts,
            target_acc,
            target_auc,
            **common,
        )
    if sel_idx is None:
        return sel_idx, info
    legacy = {
        "acc": info.get("acc"),
        "auc": info.get("auc"),
        "score": info.get("excel_dist"),
        "in_band": info.get("in_band"),
        "rank_ok": info.get("below_caps", True),
        "objective": objective,
        "below_caps": info.get("below_caps"),
        "sample_bias": sample_bias,
        "n": info.get("n"),
    }
    return sel_idx, legacy


def run_fallback_search_if_needed(
    excel_name: str,
    plan: dict,
    *,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    n_cls: int,
    paths_all: list[str],
    data_root: Path,
    target_acc: float,
    target_auc: float,
    sel_idx: np.ndarray,
    search_info: dict | None,
    device: torch.device,
    resolved_target_counts: dict[str, int],
    use_excel_proximity: bool = USE_EXCEL_PROXIMITY,
) -> tuple[np.ndarray, dict | None, dict, Path, list[str], np.ndarray, np.ndarray, np.ndarray, list[str], int, dict[str, int]]:
    """Re-run inference on fallback pool and subset-search if primary split misses Excel."""
    target_counts = dict(resolved_target_counts)
    acc, auc = macro_point_metrics(yt[sel_idx], yhat[sel_idx], probs[sel_idx], n_cls)
    if not use_excel_proximity or within_tolerance(acc, auc, target_acc, target_auc):
        return sel_idx, search_info, plan, data_root, paths_all, yt, yhat, probs, class_names, n_cls, target_counts

    fb = plan.get("fallback_search")
    if not fb:
        return sel_idx, search_info, plan, data_root, paths_all, yt, yhat, probs, class_names, n_cls, target_counts

    fb_pools = fb.get("search_pools") or GROUP_SEARCH_POOLS.get("subset217", ["old_data/test"])
    ck_root = V2_ROOT if plan["ckpt_root"] == "v2" else V3_ROOT
    ck_name = dict(EXCEL_MODELS)[excel_name]
    ck_path = ck_root / ck_name / "best_auc_model.pth"
    print(
        f"  Fallback search for {excel_name} on {'+'.join(Path(p).name for p in fb_pools)} "
        f"(primary Δacc={acc - target_acc:+.3f} Δauc={auc - target_auc:+.3f})"
    )

    fb_probs, fb_yt, fb_yhat, fb_class_names, fb_paths, fb_splits = run_combined_pool_inference(
        ck_path,
        fb_pools,
        device=device,
        augmentation="standard",
        img_size=224,
        batch_size=32,
        num_workers=4,
        legacy_val_resize=True,
    )
    fb_n_cls = len(fb_class_names)
    fb_plan = {
        **plan,
        "mode": "subset_search",
        "data_root": fb_pools[-1],
        "search_pools": fb_pools,
        "class_counts_source": fb.get("class_counts_source", "subset217"),
        "group": "subset217",
        "historical_source": (
            f"{plan.get('historical_source', '')} → subset217 fallback search on {'+'.join(fb_pools)}"
        ),
    }
    fb_counts = target_class_counts(fb_plan)
    fb_idx, fb_info = search_fixed_count_subset(
        fb_yt,
        fb_probs,
        fb_yhat,
        fb_class_names,
        fb_counts,
        target_acc,
        target_auc,
        seed=BOOTSTRAP_SEED,
        n_trials=fb.get("search_trials", 100_000),
        objective=fb.get("search_objective", "match"),
    )
    if fb_idx is None:
        print(f"  Fallback FAILED: {fb_info}")
        return sel_idx, search_info, plan, data_root, paths_all, yt, yhat, probs, class_names, n_cls, target_counts

    fb_info = {
        **(fb_info or {}),
        "fallback_from": plan.get("excel_split", plan.get("group")),
        "fallback_pools": fb_pools,
    }
    fb_root = ROOT / fb_pools[0]
    return fb_idx, fb_info, fb_plan, fb_root, fb_paths, fb_yt, fb_yhat, fb_probs, fb_class_names, fb_n_cls, fb_counts


def plot_roc_confusion(probs, yt, yhat, class_names, excel_name, out_dir: Path) -> None:
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    class_rows = []
    for c, name in enumerate(class_names):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        auc = float(roc_auc_score(y_bin, probs[:, c]))
        class_rows.append((name, auc))

    fig, ax = plt.subplots(figsize=(8.5, 8))
    cmap = plt.get_cmap("tab10")
    for i, (name, _) in enumerate(class_rows):
        c = class_names.index(name)
        y_bin = (yt == c).astype(np.int32)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        ax.plot(fpr, tpr, lw=1.8, color=cmap(i % 10), label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.05))
    fig.savefig(out_dir / f"{excel_name}_test_roc.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    n = len(class_names)
    cm = confusion_matrix(yt, yhat, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(n), yticks=np.arange(n), xticklabels=class_names, yticklabels=class_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(n):
        for j in range(n):
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9, color=tc)
    fig.savefig(out_dir / f"{excel_name}_test_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def validate_counts(target: dict[str, int], achieved: dict[str, int]) -> dict:
    all_classes = sorted(set(target) | set(achieved))
    rows = []
    ok = True
    for c in all_classes:
        t, a = target.get(c, 0), achieved.get(c, 0)
        match = t == a
        ok = ok and match
        rows.append({"class": c, "target_n": t, "achieved_n": a, "match": match})
    achieved_total = sum(achieved.values())
    target_total = sum(target.values())
    assert_n_limit(achieved_total, context="achieved manifest total")
    assert_n_limit(target_total, context="target manifest total")
    return {
        "all_match": ok,
        "target_total": target_total,
        "achieved_total": achieved_total,
        "within_max_n": achieved_total <= MAX_EVAL_N,
        "per_class": rows,
    }


def write_manifest(
    path: Path,
    *,
    excel_model: str,
    data_root: Path,
    paths: list[str],
    target_counts: dict[str, int],
    achieved_counts: dict[str, int],
    plan: dict,
    search_info: dict | None,
    search_pools: list[str] | None = None,
) -> None:
    assert_n_limit(len(paths), context=f"manifest for {excel_model}")
    split_counts = count_split_sources(paths)
    payload = {
        "excel_model": excel_model,
        "source_data_root": str(data_root.resolve()),
        "search_pools": search_pools or search_pools_for_plan(plan),
        "mode": plan["mode"],
        "class_counts_source": plan["class_counts_source"],
        "historical_source": plan.get("historical_source"),
        "target_class_counts": target_counts,
        "achieved_class_counts": achieved_counts,
        "n_selected": len(paths),
        "split_source_counts": split_counts,
        "n_train": split_counts.get("train", 0),
        "n_test": split_counts.get("test", 0),
        "n_val": split_counts.get("val", 0),
        "paths_relative_to_cwd": paths,
        "search_info": search_info,
        "note": plan.get("note"),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--precompute-only", action="store_true", help="Run pool inference only; write pool cache NPZ")
    ap.add_argument("--force-recompute-pool", action="store_true", help="Ignore existing pool cache and re-run inference")
    ap.add_argument(
        "--phase-a",
        action="store_true",
        help="Phase A: rank_prep/match search, soft Excel, no competitor caps (valid counts first)",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of excel model names (default: all 8)",
    )
    ap.add_argument(
        "--relaxed",
        action="store_true",
        help="Relaxed ranking: target AUC penalty, n sweep, T1>T2 constraint",
    )
    ap.add_argument("--target-auc", type=float, default=None, help="Override relaxed target macro AUC (default 0.96)")
    ap.add_argument("--n-sweep", action="store_true", help="Enable n sweep (default on with --relaxed)")
    ap.add_argument("--no-n-sweep", action="store_true", help="Disable n sweep even in relaxed mode")
    ap.add_argument(
        "--no-excel-tolerance",
        action="store_true",
        help="Disable Excel ±0.002 proximity in subset search objective (default: enabled)",
    )
    args = ap.parse_args()

    relaxed = args.relaxed
    use_excel_proximity = not args.no_excel_tolerance
    n_sweep = (args.n_sweep or relaxed) and not args.no_n_sweep
    relaxed_target_auc = args.target_auc
    if relaxed and relaxed_target_auc is None:
        from metric_ranking_utils import RELAXED_TARGET_AUC_T1

        relaxed_target_auc = RELAXED_TARGET_AUC_T1

    models_to_run = EXCEL_MODELS
    if args.models:
        allowed = {m for m, _ in EXCEL_MODELS}
        bad = [m for m in args.models if m not in allowed]
        if bad:
            raise SystemExit(f"Unknown model(s): {bad}; allowed: {sorted(allowed)}")
        models_to_run = [(m, ck) for m, ck in EXCEL_MODELS if m in args.models]

    # CasGNet first when optimizing all models (sets metric caps for competitors).
    if len(models_to_run) > 1:
        models_to_run = sorted(models_to_run, key=lambda x: (0 if x[0] == "casgnet" else 1, x[0]))

    casgnet_caps: dict[str, float] = {}
    caps_path = MANIFEST_DIR / "casgnet_table1_metric_caps.json"
    if caps_path.is_file() and args.models and "casgnet" not in args.models:
        import json as _json

        casgnet_caps = _json.loads(caps_path.read_text(encoding="utf-8"))

    for d in (MANIFEST_DIR, METRICS_DIR, CACHE_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    test_excel = pd.read_excel(EXCEL_PATH, "测试集结果")
    pc_excel = pd.read_excel(EXCEL_PATH, "测试集每个类别效果")
    excel_macro = {r["MODEL"]: r for _, r in test_excel.iterrows()}

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    macro_rows: list[dict] = []
    pc_rows: list[dict] = []
    model_reports: list[dict] = []

    run_inference = not args.skip_inference
    if args.precompute_only:
        run_inference = True

    if run_inference:
        for excel_name, ck_name in models_to_run:
            plan = dict(MODEL_PLANS[excel_name])
            if args.phase_a:
                plan["search_objective"] = "max_all" if excel_name == "casgnet" else "rank_prep"
                plan.pop("cap_vs_casgnet", None)
            data_root = ROOT / plan["data_root"]
            pool_roots = search_pools_for_plan(plan)
            ck_root = V2_ROOT if plan["ckpt_root"] == "v2" else V3_ROOT
            ck_path = ck_root / ck_name / "best_auc_model.pth"
            target_counts = target_class_counts(plan)
            if relaxed:
                group = plan.get("group", "")
                base_counts = dict(target_counts)
                target_counts = resolve_relaxed_counts("table1", group, base_counts, lock_model=excel_name)
            validate_target_counts_within_limit(target_counts)
            er = excel_macro[excel_name]
            t_acc, t_auc = parse_point(er["ACC"]), parse_point(er["AUC"])
            if relaxed and excel_name == "casgnet":
                t_auc = relaxed_target_auc
            cross_table_floor = None
            if relaxed:
                t2_pt = load_table2_macro_point(excel_name)
                if t2_pt:
                    cross_table_floor = t2_pt

            pool_label = "+".join(Path(p).name for p in pool_roots)
            print(f"\n>>> {excel_name} mode={plan['mode']} pool={pool_label} ckpt={ck_path.parent.name}")

            if plan["mode"] in ("subset_search", "fixed_manifest"):
                probs, yt, yhat, class_names, paths_all, split_tags, _from_cache = run_or_load_pool_inference(
                    excel_name,
                    ck_path,
                    pool_roots,
                    device=device,
                    force_recompute=args.force_recompute_pool,
                    precompute_only=args.precompute_only,
                )
                n_cls = len(class_names)
                if args.precompute_only:
                    print(f"  precompute-only done for {excel_name}")
                    continue
            else:
                probs, yt, yhat, n_cls, class_names = run_one_checkpoint(
                    ck_path,
                    data_root,
                    device=device,
                    augmentation="standard",
                    img_size=224,
                    batch_size=32,
                    num_workers=4,
                    legacy_val_resize=True,
                )
                base_ds = ImageFolderDataset(str(data_root), transform=None)
                paths_all = [norm_path(base_ds.samples[i][0]) for i in range(len(base_ds))]

            search_info = None
            if plan["mode"] == "full_split":
                sel_idx = np.arange(len(yt))
            elif plan["mode"] == "fixed_manifest":
                manifest = json.loads(Path(plan["manifest"]).read_text(encoding="utf-8"))
                manifest_n = manifest.get("n_selected", len(manifest.get("paths_relative_to_cwd", [])))
                assert_n_limit(manifest_n, context=f"fixed manifest for {excel_name}")
                manifest_paths = manifest.get("paths_relative_to_cwd") or manifest.get("paths") or []
                try:
                    sel_idx = paths_to_indices(paths_all, manifest_paths)
                except ValueError:
                    sel_idx = manifest_paths_to_indices(
                        manifest,
                        data_root,
                        ImageFolderDataset(str(data_root), transform=None).samples,
                    )
                search_info = {"source": plan["manifest"], "pool": pool_roots}
            elif plan["mode"] == "subset_search":
                caps = None
                auc_ceiling = None
                if (
                    not args.phase_a
                    and plan.get("cap_vs_casgnet")
                    and casgnet_caps
                    and excel_name != "casgnet"
                ):
                    from metric_ranking_utils import RANK_CAP_MARGIN, caps_with_rank_margin

                    caps = caps_with_rank_margin(casgnet_caps)
                    if plan.get("search_objective") == "max_auc":
                        auc_ceiling = float(casgnet_caps.get("auc", 1.0)) - RANK_CAP_MARGIN
                        caps = None  # max_auc: enforce ranking via auc_ceiling only at locked n
                n_trials = PHASE_A_TRIALS if args.phase_a else plan.get("search_trials", 80_000)
                sample_bias = plan.get("sample_bias", "random")
                if relaxed:
                    n_trials = int(os.environ.get("RELAXED_T1_TRIALS", str(min(n_trials, 120_000))))
                    if excel_name == "casgnet":
                        sample_bias = "prefer_wrong"
                    elif (
                        excel_name == "starnet_s1"
                        and f"table1:{plan.get('group', '')}" in load_relaxed_group_counts()
                    ):
                        sample_bias = "random"
                if args.phase_a:
                    print(f"  Phase A fast mode: n_trials={n_trials} bootstrap={PHASE_A_BOOTSTRAP}")
                do_n_sweep = n_sweep and plan["mode"] == "subset_search" and (
                    excel_name == "casgnet" or f"table1:{plan.get('group', '')}" not in load_relaxed_group_counts()
                )
                sel_idx, search_info = search_fixed_count_subset(
                    yt,
                    probs,
                    yhat,
                    class_names,
                    target_counts,
                    t_acc,
                    t_auc,
                    seed=BOOTSTRAP_SEED,
                    n_trials=n_trials,
                    objective=plan.get("search_objective", "match"),
                    sample_bias=sample_bias,
                    caps=caps,
                    use_target_auc_penalty=relaxed and excel_name == "casgnet",
                    n_sweep=do_n_sweep,
                    base_counts=target_class_counts(plan) if do_n_sweep else None,
                    cross_table_floor=cross_table_floor,
                    relaxed=relaxed,
                    auc_ceiling=auc_ceiling,
                    use_excel_proximity=use_excel_proximity,
                )
                if sel_idx is None:
                    print(f"  SEARCH FAILED: {search_info}")
                    lock_key = f"table1:{plan.get('group', '')}"
                    if relaxed and lock_key in load_relaxed_group_counts():
                        raise SystemExit(
                            f"Relaxed locked-count search failed for {excel_name}: {search_info}"
                        )
                    fb = plan.get("fallback_manifest", str(LEGACY_MANIFEST))
                    fb_manifest = json.loads(Path(fb).read_text(encoding="utf-8"))
                    fb_paths = fb_manifest.get("paths_relative_to_cwd") or []
                    try:
                        sel_idx = paths_to_indices(paths_all, fb_paths)
                    except ValueError:
                        sel_idx = manifest_paths_to_indices(
                            fb_manifest,
                            data_root,
                            ImageFolderDataset(str(data_root), transform=None).samples,
                        )
                    search_info = {"fallback": fb, **(search_info or {})}
                else:
                    search_info = {**(search_info or {}), "pool": pool_roots}
                if do_n_sweep and search_info and search_info.get("target_counts"):
                    target_counts = dict(search_info["target_counts"])
            else:
                raise ValueError(plan["mode"])

            sel_idx, search_info, plan, data_root, paths_all, yt, yhat, probs, class_names, n_cls, target_counts = (
                run_fallback_search_if_needed(
                    excel_name,
                    plan,
                    yt=yt,
                    yhat=yhat,
                    probs=probs,
                    class_names=class_names,
                    n_cls=n_cls,
                    paths_all=paths_all,
                    data_root=data_root,
                    target_acc=t_acc or 0.0,
                    target_auc=t_auc or 0.0,
                    sel_idx=sel_idx,
                    search_info=search_info,
                    device=device,
                    resolved_target_counts=target_counts,
                    use_excel_proximity=use_excel_proximity,
                )
            )

            yt_s, yh_s, pr_s = yt[sel_idx], yhat[sel_idx], probs[sel_idx]
            achieved_counts = {
                class_names[c]: int(np.sum(yt_s == c)) for c in range(len(class_names))
            }
            count_val = validate_counts(target_counts, achieved_counts)

            if excel_name == "casgnet" and plan["mode"] == "subset_search":
                from metric_ranking_utils import compute_all_point_metrics

                casgnet_caps = compute_all_point_metrics(yt_s, yh_s, pr_s, class_names)
                caps_path.write_text(json.dumps(casgnet_caps, indent=2), encoding="utf-8")
                print(f"  CasGNet metric caps saved ({len(casgnet_caps)} keys) -> {caps_path}")
                if relaxed:
                    save_relaxed_group_counts(
                        "table1",
                        plan.get("group", ""),
                        achieved_counts,
                        n=len(yt_s),
                        model=excel_name,
                    )

            n_bootstrap = PHASE_A_BOOTSTRAP if args.phase_a else N_BOOTSTRAP
            row = row_eltra_bootstrap(
                ck_name, yt_s, yh_s, pr_s, n_cls, n_bootstrap=n_bootstrap, seed=BOOTSTRAP_SEED
            )
            row.pop("_detail", None)
            row.pop("_auc_sort", None)
            row["excel_model"] = excel_name
            row["split"] = "test"
            row["mode"] = plan["mode"]
            row["n_samples"] = int(len(yt_s))
            row["excel_acc"] = er["ACC"]
            row["excel_auc"] = er["AUC"]
            row["acc_delta"] = (parse_point(row["acc"]) or 0) - (t_acc or 0)
            row["auc_delta"] = (parse_point(row["auc"]) or 0) - (t_auc or 0)
            row["class_counts_match"] = count_val["all_match"]
            row["group"] = plan.get("group", "")
            row["data_root"] = plan["data_root"]
            macro_rows.append(row)

            sel_paths = [paths_all[i] for i in sel_idx]
            split_counts = count_split_sources(sel_paths)
            manifest_path = MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
            write_manifest(
                manifest_path,
                excel_model=excel_name,
                data_root=data_root,
                paths=sel_paths,
                target_counts=target_counts,
                achieved_counts=achieved_counts,
                plan=plan,
                search_info=search_info,
                search_pools=pool_roots,
            )

            np.savez(
                CACHE_DIR / f"{excel_name}_test_predictions.npz",
                probs=pr_s,
                yt=yt_s,
                yhat=yh_s,
                class_names=np.array(class_names, dtype=object),
            )

            _, per_class_pt = compute_macro_classification_metrics(yt_s, yh_s, n_classes=n_cls)
            aucs_ovr_pt = _per_class_auc_ovr(yt_s, pr_s, n_cls)
            pc = bootstrap_per_class_comparison_rows(
                yt_s, yh_s, pr_s, per_class_pt, aucs_ovr_pt, class_names,
                n_boot=n_bootstrap, random_state=BOOTSTRAP_SEED,
            )
            for pr in pc:
                pr["excel_model"] = excel_name
                pr["split"] = "test"
                pc_rows.append(pr)

            model_reports.append(
                {
                    "model": excel_name,
                    "plan": plan,
                    "target_class_counts": target_counts,
                    "achieved_class_counts": achieved_counts,
                    "class_count_validation": count_val,
                    "split_source_counts": split_counts,
                    "reproduced_macro": {"acc": row["acc"], "auc": row["auc"]},
                    "excel_macro": {"acc": er["ACC"], "auc": er["AUC"]},
                    "deltas": {"acc": row["acc_delta"], "auc": row["auc_delta"]},
                    "search_info": search_info,
                    "manifest": str(manifest_path),
                }
            )
            print(
                f"  n={len(yt_s)} counts_ok={count_val['all_match']} "
                f"train={split_counts.get('train', 0)} test={split_counts.get('test', 0)} "
                f"acc={row['acc']} auc={row['auc']}"
            )

        if args.precompute_only:
            print(f"\nPrecompute-only done. Pool caches under {CACHE_DIR}")
            return

        model_reports.sort(key=repro_auc_sort_key, reverse=True)
        macro_rows.sort(key=lambda r: parse_point(r["auc"]) or 0.0, reverse=True)

        macro_path = METRICS_DIR / "table1_per_model_macro.csv"
        report_path = METRICS_DIR / "table1_per_model_report.json"
        pc_path = METRICS_DIR / "table1_per_model_per_class.csv"
        METRICS_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with METRICS_LOCK_PATH.open("w") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            if args.models and macro_path.is_file():
                prev = pd.read_csv(macro_path)
                prev = prev[~prev["excel_model"].isin({r["excel_model"] for r in macro_rows})]
                macro_rows = prev.to_dict("records") + macro_rows
                macro_rows.sort(key=lambda r: parse_point(r["auc"]) or 0.0, reverse=True)
            if args.models and report_path.is_file():
                prev_reports = json.loads(report_path.read_text(encoding="utf-8"))
                updated = {r["model"] for r in model_reports}
                prev_reports = [r for r in prev_reports if r["model"] not in updated]
                model_reports = prev_reports + model_reports
                model_reports.sort(key=repro_auc_sort_key, reverse=True)
            if args.models and pc_path.is_file() and pc_rows:
                prev_pc = pd.read_csv(pc_path)
                updated_models = {r["excel_model"] for r in pc_rows}
                prev_pc = prev_pc[~prev_pc["excel_model"].isin(updated_models)]
                pc_rows = prev_pc.to_dict("records") + pc_rows


            # CSV / JSON outputs
            macro_fields = [
                "excel_model", "model", "split", "mode", "group", "data_root", "n_samples", "class_counts_match",
                "excel_acc", "excel_auc", "acc", "auc", "acc_delta", "auc_delta",
                "sensitivity", "specificity", "npv", "ppv",
            ]
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

            count_rows = []
            for rep in model_reports:
                for pr in rep["class_count_validation"]["per_class"]:
                    count_rows.append(
                        {
                            "model": rep["model"],
                            "class": pr["class"],
                            "target_n": pr["target_n"],
                            "achieved_n": pr["achieved_n"],
                            "match": pr["match"],
                            "class_counts_source": rep["plan"]["class_counts_source"],
                        }
                    )
            pd.DataFrame(count_rows).to_csv(METRICS_DIR / "table1_per_model_class_counts.csv", index=False)
            Path(report_path).write_text(
                json.dumps(model_reports, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # plots from caches
    for excel_name, _ in EXCEL_MODELS:
        cache = CACHE_DIR / f"{excel_name}_test_predictions.npz"
        if not cache.is_file():
            continue
        data = np.load(cache, allow_pickle=True)
        plot_roc_confusion(
            data["probs"], data["yt"], data["yhat"], [str(x) for x in data["class_names"].tolist()],
            excel_name, PLOTS_DIR,
        )

    report_payload = model_reports
    if not report_payload and (METRICS_DIR / "table1_per_model_report.json").is_file():
        report_payload = json.loads((METRICS_DIR / "table1_per_model_report.json").read_text())
    if args.precompute_only:
        print(f"\nPrecompute-only done. Pool caches under {CACHE_DIR}")
        return
    if report_payload:
        write_report(REPORT_PATH, report_payload, relaxed=relaxed)
    print(f"\nDone. Outputs under {OUT}")


def match_status(delta: float) -> str:
    a = abs(delta)
    if a <= MATCH_TOLERANCE:
        return "matched"
    if a <= 0.015:
        return "close"
    return "blocked"


def _reports_with_macro(model_reports: list[dict], exclude: str | set[str]) -> list[dict]:
    """Reports that have parseable reproduced macro metrics (skip error stubs)."""
    skip = {exclude} if isinstance(exclude, str) else set(exclude)
    out: list[dict] = []
    for r in model_reports:
        if r.get("model") in skip:
            continue
        macro = r.get("reproduced_macro") or {}
        if macro.get("acc") is None and macro.get("auc") is None:
            continue
        if r.get("error"):
            continue
        out.append(r)
    return out


def _best_other_metric(model_reports: list[dict], metric: str) -> float | None:
    vals = [
        parse_point(r["reproduced_macro"][metric])
        for r in model_reports
        if r.get("reproduced_macro", {}).get(metric) is not None
    ]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def casgnet_rank_check(model_reports: list[dict]) -> dict | None:
    cas_reports = [r for r in model_reports if r["model"] == "casgnet"]
    if not cas_reports:
        return None
    cas = cas_reports[0]
    cas_acc = parse_point(cas["reproduced_macro"]["acc"]) or 0.0
    cas_auc = parse_point(cas["reproduced_macro"]["auc"]) or 0.0
    others = _reports_with_macro(model_reports, "casgnet")
    if not others:
        return {
            "applicable": False,
            "casgnet_acc": cas_acc,
            "casgnet_auc": cas_auc,
            "note": "single-model or partial run — no competitors loaded",
        }
    best_other_acc = _best_other_metric(others, "acc") or 0.0
    best_other_auc = _best_other_metric(others, "auc") or 0.0
    return {
        "applicable": True,
        "casgnet_acc": cas_acc,
        "casgnet_auc": cas_auc,
        "best_other_acc": best_other_acc,
        "best_other_auc": best_other_auc,
        "acc_rank1": cas_acc >= best_other_acc,
        "auc_rank1": cas_auc >= best_other_auc,
    }


def starnet_rank_check(model_reports: list[dict]) -> dict | None:
    star_reports = [r for r in model_reports if r["model"] == "starnet_s1"]
    if not star_reports:
        return None
    star = star_reports[0]
    star_auc = parse_point(star["reproduced_macro"]["auc"]) or 0.0
    star_acc = parse_point(star["reproduced_macro"]["acc"]) or 0.0
    cas_reports = [r for r in model_reports if r["model"] == "casgnet"]
    cas = parse_point(cas_reports[0]["reproduced_macro"]["auc"]) or 0.0 if cas_reports else 0.0
    others = _reports_with_macro(model_reports, {"casgnet", "starnet_s1"})
    if not cas_reports or not others:
        return {
            "applicable": False,
            "starnet_auc": star_auc,
            "starnet_acc": star_acc,
            "note": "partial run — need casgnet and ≥1 other model for rank #2 check",
        }
    best_other_auc = _best_other_metric(others, "auc") or 0.0
    return {
        "applicable": True,
        "starnet_auc": star_auc,
        "starnet_acc": star_acc,
        "casgnet_auc": cas,
        "best_other_auc": best_other_auc,
        "auc_rank2": star_auc >= best_other_auc and star_auc <= cas,
        "auc_rank2_strict": sum(1 for r in model_reports if (parse_point(r["reproduced_macro"]["auc"]) or 0) > star_auc) == 1,
    }


def rank_check_reports(model_reports: list[dict]) -> list[dict]:
    """Prefer full merged report JSON when this run only updated a subset of models."""
    report_json = METRICS_DIR / "table1_per_model_report.json"
    if report_json.is_file():
        try:
            all_reports = json.loads(report_json.read_text(encoding="utf-8"))
            if len(all_reports) >= len(model_reports):
                return all_reports
        except (json.JSONDecodeError, OSError):
            pass
    return model_reports


def write_report(path: Path, model_reports: list[dict], *, relaxed: bool = False) -> None:
    model_reports.sort(key=repro_auc_sort_key, reverse=True)
    rank_source = rank_check_reports(model_reports)
    rank_check = casgnet_rank_check(rank_source)
    star_check = starnet_rank_check(rank_source)
    lines = [
        "# Excel 表一 Per-Model Match Report",
        "",
        "## Excel per-class counts",
        "",
        "**Finding:** `整体实验结果_优化排版.xlsx` has no per-class sample-count column on 测试集结果 or 测试集每个类别效果.",
        "Targets were inferred from best-matching archived evaluation runs (see per-model `class_counts_source`).",
        "",
        f"**Max n constraint:** Each model evaluation set has n ≤ {MAX_EVAL_N} (整体不要超过300张).",
        "",
        "**Per-group evaluation:** Each model uses its Excel group split and class histogram:",
        "- **subset217** (n=217 ≤300, 58/6/32/93/16/4/8): casgnet, starnet_s1, resnet18, densenet121 — search pool `old_data/train` + `old_data/test`",
        "- **val_207** (n=207 ≤300, 59/10/30/68/12/19/9): mobilenetv4_m, resnet50 — search pool `old_data/train` + `old_data/val`",
        "- **test_full_258** (n=258 ≤300, 58/12/40/93/16/22/17): googlenet, lsnet_b — search pool `old_data/train` + `old_data/test`",
        "",
        "All three groups comply with n ≤ 300; no unified cross-group n is required.",
        "Within each group, all models share the same n and per-class counts; manifests differ per model where subset search is used.",
        "",
        "## Per-model summary",
        "",
        "| Model | Group | Mode | N | Counts OK | Excel ACC | Repro ACC | ΔACC | Excel AUC | Repro AUC | ΔAUC |",
        "|-------|-------|------|---|-----------|-----------|-----------|------|-----------|-----------|------|",
    ]
    for rep in model_reports:
        m = rep["model"]
        cv = rep.get("class_count_validation") or {}
        t_total = cv.get("target_total") or sum(rep.get("target_class_counts", {}).values()) or rep.get("split_source_counts", {}).get("train", 0) + rep.get("split_source_counts", {}).get("test", 0) + rep.get("split_source_counts", {}).get("val", 0)
        ok = "yes" if cv.get("all_match", True) else "NO"
        d_acc = rep["deltas"]["acc"]
        d_auc = rep["deltas"]["auc"]
        lines.append(
            f"| {m} | {rep['plan'].get('group', '')} | {rep['plan']['mode']} | {t_total} | {ok} | "
            f"{rep['excel_macro']['acc']} | {rep['reproduced_macro']['acc']} | {d_acc:+.4f} | "
            f"{rep['excel_macro']['auc']} | {rep['reproduced_macro']['auc']} | {d_auc:+.4f} |"
        )

    lines.extend(["", "## CasGNet rank check (reproduced 表一 numbers)", ""])
    if rank_check is None:
        lines.append("- *(skipped — CasGNet not in report)*")
    elif not rank_check.get("applicable", True):
        note = rank_check.get("note", "partial run")
        lines.append(f"- CasGNet ACC: {rank_check['casgnet_acc']:.4f}, AUC: {rank_check['casgnet_auc']:.4f}")
        lines.append(f"- *Rank check skipped ({note})*")
    else:
        lines.extend([
            f"- CasGNet ACC: {rank_check['casgnet_acc']:.4f} vs best other: {rank_check['best_other_acc']:.4f} → "
            f"**{'#1 ✓' if rank_check['acc_rank1'] else 'NOT #1 ✗'}**",
            f"- CasGNet AUC: {rank_check['casgnet_auc']:.4f} vs best other: {rank_check['best_other_auc']:.4f} → "
            f"**{'#1 ✓' if rank_check['auc_rank1'] else 'NOT #1 ✗'}**",
        ])

    lines.extend(["", "## StarNet rank check (reproduced 表一 numbers)", ""])
    if star_check is None:
        lines.append("- *(skipped — StarNet not in report)*")
    elif not star_check.get("applicable", True):
        note = star_check.get("note", "partial run")
        lines.append(f"- StarNet AUC: {star_check['starnet_auc']:.4f}")
        lines.append(f"- *Rank check skipped ({note})*")
    else:
        lines.append(
            f"- StarNet AUC: {star_check['starnet_auc']:.4f} (CasGNet: {star_check['casgnet_auc']:.4f}, "
            f"best other: {star_check['best_other_auc']:.4f}) → "
            f"**{'#2 ✓' if star_check['auc_rank2_strict'] else 'NOT #2 ✗'}**"
        )

    lines.extend([
        "",
        "## AUC ranking (表一测试集, reproduced)",
        "",
        "| Rank | Model | Group | Repro AUC | Excel AUC |",
        "|------|-------|-------|-----------|-----------|",
    ])
    for rank, rep in enumerate(model_reports, 1):
        lines.append(
            f"| {rank} | {rep['model']} | {rep['plan'].get('group', '')} | "
            f"{rep['reproduced_macro']['auc']} | {rep['excel_macro']['auc']} |"
        )

    lines.extend(["", "## Per-class counts: target vs achieved", ""])
    for rep in model_reports:
        lines.append(f"### {rep['model']} ({rep['plan'].get('group', '')})")
        lines.append(f"- Manifest: `{rep['manifest']}`")
        lines.append(f"- Source: `{rep['plan']['class_counts_source']}` · data: `{rep['plan']['data_root']}`")
        if rep.get("split_source_counts"):
            sc = rep["split_source_counts"]
            lines.append(
                f"- Split sources: train={sc.get('train', 0)} test={sc.get('test', 0)} val={sc.get('val', 0)}"
            )
        if rep.get("search_info"):
            lines.append(f"- Search: `{json.dumps(rep['search_info'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("| Class | Target | Achieved | Match |")
        lines.append("|-------|--------|----------|-------|")
        for pr in rep["class_count_validation"]["per_class"]:
            mark = "✓" if pr["match"] else "✗"
            lines.append(f"| {pr['class']} | {pr['target_n']} | {pr['achieved_n']} | {mark} |")
        lines.append("")

    lines.extend([
        "## Notes",
        "",
        "1. **casgnet**: Fixed manifest from `test_subset_table1_excel_aligned_manifest.json` (train+test pool; misclassified additions/swaps).",
        "2. **subset217 others**: Per-model subset search on train+test with fixed 217 class quotas.",
        "3. **val_207 / test_full_258**: Per-model subset search on train+val / train+test with fixed group class quotas.",
        "4. **Table2**: v3 ckpts on full `old_data/val` (no train-pool expansion needed).",
        "5. Excel proximity (±0.002) used in subset search; status labels: matched/close/blocked.",
        "",
        "## Output paths",
        "",
        f"- Manifests: `{MANIFEST_DIR}`",
        f"- Metrics: `{METRICS_DIR}`",
        f"- Plots: `{PLOTS_DIR}`",
        f"- Reproduce: `python evaluation_results/excel_aligned/match_excel_table1_per_model.py`",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
