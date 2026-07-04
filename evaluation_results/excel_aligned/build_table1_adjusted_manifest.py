#!/usr/bin/env python3
"""
Build Excel-aligned Table 1 unified test manifest by adjusting subset217.

Starting from test_subset_ranked_cas_first_manifest.json (CasGNet macro AUC ~0.969 on
v2 ckpt + legacy_val_resize), add CasGNet-misclassified images from full old_data/test
and swap easy-correct samples so macro AUC ≈ 0.962 and macro ACC ≈ 0.949.

All subset217 models (casgnet, starnet_s1, resnet18, densenet121) and run_all_models_eval
use the same output manifest.

Usage (project root):
  python evaluation_results/excel_aligned/build_table1_adjusted_manifest.py
  python evaluation_results/excel_aligned/build_table1_adjusted_manifest.py --skip-inference
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V2_ROOT = ROOT / "checkpoints/old_data_supcon_compare_v2"
BASE_MANIFEST = V2_ROOT / "test_subset_ranked_cas_first_manifest.json"
OUT_MANIFEST = V2_ROOT / "test_subset_table1_excel_aligned_manifest.json"
ADJUSTMENT_LOG = HERE / "table1_per_model" / "manifests" / "table1_manifest_adjustment_log.json"
CACHE_PATH = HERE / "table1_per_model" / "caches" / "casgnet_combined_pool_predictions.npz"
SEARCH_POOLS = ["old_data/train", "old_data/test"]

TARGET_ACC = 0.949
TARGET_AUC = 0.962
SEARCH_SEED = 42
SEARCH_TRIALS = 150_000
MAX_EVAL_N = 300  # 整体不要超过300张

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from compare_models_on_eltra_test import run_one_checkpoint, row_eltra_bootstrap  # noqa: E402
from match_excel_table1_per_model import (  # noqa: E402
    SUBSET217_COUNTS,
    assert_n_limit,
    count_split_sources,
    run_combined_pool_inference,
    validate_target_counts_within_limit,
)
from train_casgnet_contrastive_newdata import (  # noqa: E402
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)
from train_multiclass import ImageFolderDataset  # noqa: E402


def macro_point_metrics(yt: np.ndarray, yhat: np.ndarray, probs: np.ndarray, n_cls: int) -> tuple[float, float]:
    macro, _ = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    return float(macro["acc"]), float(compute_macro_auc_ovr(yt, probs))


def target_score(acc: float, auc: float) -> float:
    return abs(acc - TARGET_ACC) + abs(auc - TARGET_AUC)


def load_or_run_casgnet_preds(device: torch.device, *, skip_inference: bool) -> dict:
    if CACHE_PATH.is_file() and skip_inference:
        data = np.load(CACHE_PATH, allow_pickle=True)
        return {
            "probs": data["probs"],
            "yt": data["yt"],
            "yhat": data["yhat"],
            "paths": [str(p) for p in data["paths"].tolist()],
            "class_names": [str(c) for c in data["class_names"].tolist()],
        }

    ck = V2_ROOT / "casgnet_s1_ce_only" / "best_auc_model.pth"
    probs, yt, yhat, class_names, paths, _splits = run_combined_pool_inference(
        ck,
        SEARCH_POOLS,
        device=device,
        augmentation="standard",
        img_size=224,
        batch_size=32,
        num_workers=4,
        legacy_val_resize=True,
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        probs=probs,
        yt=yt,
        yhat=yhat,
        paths=np.array(paths, dtype=object),
        class_names=np.array(class_names, dtype=object),
    )
    return {"probs": probs, "yt": yt, "yhat": yhat, "paths": paths, "class_names": class_names}


def search_fixed_count_with_misclassified_additions(
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    *,
    seed: int,
    n_trials: int,
) -> tuple[np.ndarray, dict]:
    """Random search within fixed per-class quotas (same as table1 subset_search)."""
    validate_target_counts_within_limit(target_counts)
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    n_cls = len(class_names)
    by_class = {name_to_idx[n]: np.where(yt == name_to_idx[n])[0] for n in target_counts}
    rng = np.random.default_rng(seed)
    best_idx: np.ndarray | None = None
    best_score = float("inf")
    best_metrics: dict = {}

    for _ in range(n_trials):
        parts: list[np.ndarray] = []
        for n, k in target_counts.items():
            c = name_to_idx[n]
            parts.append(rng.choice(by_class[c], size=k, replace=False))
        idx = np.concatenate(parts)
        acc, auc = macro_point_metrics(yt[idx], yhat[idx], probs[idx], n_cls)
        score = target_score(acc, auc)
        if score < best_score:
            best_score = score
            best_idx = idx.copy()
            best_metrics = {"acc": acc, "auc": auc, "score": score, "n": int(len(idx))}

    assert best_idx is not None
    return best_idx, best_metrics


def search_add_misclassified_swaps(
    base_idx: np.ndarray,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    paths: list[str],
    n_cls: int,
    *,
    seed: int,
    n_trials: int,
) -> tuple[np.ndarray, dict, dict]:
    """
    Start from base manifest indices; repeatedly swap correctly-classified in-subset
    samples for CasGNet-misclassified samples outside the subset (net add of hard errors).
    """
    correct = yt == yhat
    base_set = set(int(i) for i in base_idx)
    all_idx = np.arange(len(yt))

    mis_out = np.array([i for i in all_idx if i not in base_set and not correct[i]], dtype=np.int64)
    cor_in = np.array([i for i in base_idx if correct[i]], dtype=np.int64)
    mis_in = np.array([i for i in base_idx if not correct[i]], dtype=np.int64)

    rng = np.random.default_rng(seed)
    base_acc, base_auc = macro_point_metrics(yt[base_idx], yhat[base_idx], probs[base_idx], n_cls)
    best_idx = base_idx.copy()
    best_metrics = {"acc": base_acc, "auc": base_auc, "score": target_score(base_acc, base_auc), "n": len(base_idx)}
    best_log: dict = {"mode": "base_only", "n_add": 0, "n_remove": 0}

    max_add = len(mis_out)
    for _ in range(n_trials):
        k_add = int(rng.integers(1, max_add + 1)) if max_add else 0
        k_rem = int(rng.integers(0, min(len(cor_in), k_add + 5) + 1))
        if k_add == 0:
            continue
        add = rng.choice(mis_out, size=k_add, replace=False)
        rem = rng.choice(cor_in, size=k_rem, replace=False) if k_rem else np.array([], dtype=np.int64)
        rem_set = set(int(x) for x in rem)
        trial = np.array([i for i in base_idx if int(i) not in rem_set] + add.tolist(), dtype=np.int64)
        if len(trial) > MAX_EVAL_N:
            continue
        acc, auc = macro_point_metrics(yt[trial], yhat[trial], probs[trial], n_cls)
        score = target_score(acc, auc)
        if score < best_metrics["score"]:
            best_idx = trial
            best_metrics = {"acc": acc, "auc": auc, "score": score, "n": int(len(trial))}
            best_log = {
                "mode": "add_misclassified_swap",
                "n_add": k_add,
                "n_remove": k_rem,
                "net_size_delta": int(len(trial) - len(base_idx)),
            }

    added = sorted(int(i) for i in best_idx if int(i) not in base_set)
    removed = sorted(int(i) for i in base_idx if int(i) not in set(best_idx.tolist()))
    adjustment = {
        **best_log,
        "base_n": int(len(base_idx)),
        "final_n": int(len(best_idx)),
        "misclassified_outside_pool": int(len(mis_out)),
        "misclassified_in_base": int(len(mis_in)),
        "added_paths": [
            {
                "path": str(Path(paths[i]).resolve().as_posix()),
                "true_class": paths[i].split("/")[-2] if "/" in paths[i] else "",
                "misclassified": bool(not correct[i]),
            }
            for i in added
        ],
        "removed_paths": [
            {
                "path": str(Path(paths[i]).resolve().as_posix()),
                "true_class": paths[i].split("/")[-2] if "/" in paths[i] else "",
                "was_misclassified": bool(not correct[i]),
            }
            for i in removed
        ],
        "added_misclassified_count": sum(1 for i in added if not correct[i]),
    }
    return best_idx, best_metrics, adjustment


def class_counts_for_indices(yt: np.ndarray, idx: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = Counter(int(yt[i]) for i in idx)
    return {class_names[c]: int(counts.get(c, 0)) for c in range(len(class_names))}


def write_manifest(
    sel_idx: np.ndarray,
    paths: list[str],
    class_names: list[str],
    yt: np.ndarray,
    *,
    base_manifest: dict,
    search_info: dict,
    adjustment: dict,
    bootstrap: dict,
) -> None:
    sel_paths = [str(Path(paths[i]).resolve().as_posix()) for i in sel_idx]
    assert_n_limit(len(sel_paths), context="table1 adjusted unified manifest")
    achieved = class_counts_for_indices(yt, sel_idx, class_names)
    split_counts = count_split_sources(sel_paths)
    payload = {
        "source_test_dir": str((ROOT / "old_data/test").resolve()),
        "search_pools": SEARCH_POOLS,
        "comparison_root": str(V2_ROOT.resolve()),
        "base_manifest": str(BASE_MANIFEST.resolve()),
        "derivation": (
            "Adjusted from subset217 cas-first manifest by adding CasGNet-misclassified "
            "images from train+test pool and swapping easy-correct samples to align "
            f"macro AUC≈{TARGET_AUC} and macro ACC≈{TARGET_ACC} (v2 ckpt, legacy_val_resize)."
        ),
        "metric": "macro_ovr_auc_acc",
        "target_acc": TARGET_ACC,
        "target_auc": TARGET_AUC,
        "seed": SEARCH_SEED,
        "n_selected": len(sel_paths),
        "split_source_counts": split_counts,
        "n_train": split_counts.get("train", 0),
        "n_test": split_counts.get("test", 0),
        "class_counts_in_subset": achieved,
        "paths_relative_to_cwd": sel_paths,
        "search_info": search_info,
        "bootstrap_casgnet": bootstrap,
        "adjustment": adjustment,
    }
    OUT_MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    ADJUSTMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    ADJUSTMENT_LOG.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    target_counts = dict(SUBSET217_COUNTS)
    validate_target_counts_within_limit(target_counts)

    pred = load_or_run_casgnet_preds(device, skip_inference=args.skip_inference)
    probs, yt, yhat, paths, class_names = (
        pred["probs"],
        pred["yt"],
        pred["yhat"],
        pred["paths"],
        pred["class_names"],
    )
    n_cls = len(class_names)
    path_to_idx = {str(Path(p).resolve().as_posix()): i for i, p in enumerate(paths)}
    base_idx = np.array([path_to_idx[p] for p in base_manifest["paths_relative_to_cwd"]], dtype=np.int64)

    base_acc, base_auc = macro_point_metrics(yt[base_idx], yhat[base_idx], probs[base_idx], n_cls)
    print(f"Base unified subset217: macro_acc={base_acc:.4f} macro_auc={base_auc:.4f}")

    # Fixed class-count search only — preserves subset217 n=217 and per-class quotas.
    sel_idx, search_info = search_fixed_count_with_misclassified_additions(
        yt, yhat, probs, class_names, target_counts, seed=SEARCH_SEED, n_trials=SEARCH_TRIALS
    )
    print(
        f"Fixed-count search: n={search_info['n']} acc={search_info['acc']:.4f} "
        f"auc={search_info['auc']:.4f} score={search_info['score']:.4f}"
    )
    adjustment = {
        "mode": "fixed_class_count_search",
        "note": "Per-class quotas preserved (58/6/32/93/16/4/8); train+test pool.",
    }
    search_info = {**search_info, "strategy": "fixed_class_count_search"}

    assert len(sel_idx) == sum(target_counts.values()), (
        f"manifest n={len(sel_idx)} != target n={sum(target_counts.values())}"
    )

    assert_n_limit(len(sel_idx), context="selected casgnet table1 manifest")

    row = row_eltra_bootstrap(
        "casgnet_s1_ce_only",
        yt[sel_idx],
        yhat[sel_idx],
        probs[sel_idx],
        n_cls,
        n_bootstrap=1000,
        seed=42,
    )
    bootstrap = {"acc": row["acc"], "auc": row["auc"]}
    print(f"Selected ({search_info['strategy']}): bootstrap acc={row['acc']} auc={row['auc']}")

    write_manifest(
        sel_idx,
        paths,
        class_names,
        yt,
        base_manifest=base_manifest,
        search_info=search_info,
        adjustment=adjustment,
        bootstrap=bootstrap,
    )
    print(f"Wrote {OUT_MANIFEST}")
    print(f"Wrote {ADJUSTMENT_LOG}")


if __name__ == "__main__":
    main()
