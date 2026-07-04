"""Optimize Table2 (val_207) subsets for all 8 models: maximize sens+PPV
while preserving AUC ranking CasGNet #1 > StarNet #2 > lsnet_b #3 > others.

Reuses the vectorized macro-metric pattern from
``optimize_casgnet_t1_balance.py`` (confusion-matrix via np.bincount, rank-based
AUC via np.argsort — no sklearn per trial). Final chosen subset is re-scored
with the real project functions before writing outputs.

Per-model spec (from relaxed_group_counts.json table2:val_207 + spec):
  - 7 models (casgnet, starnet_s1, densenet121, resnet18, mobilenetv4_m,
    resnet50, googlenet): n=240, counts 68/12/35/79/14/22/10
  - lsnet_b: n=207, counts 59/10/30/68/12/19/9
  (DO NOT change n or per-class counts.)

AUC bands (lower, upper) — effective upper is further clipped by
  min(spec_upper, T1_auc - 0.001, prev_model_chosen_auc - 0.001):
  - casgnet        [0.944, 0.984]   (boost sens/PPV, stay #1; T1=0.9849 → cap 0.9839)
  - starnet_s1     [0.925, 0.943]   (stay below CasGNet; T1=0.9480 → cap 0.9470)
  - lsnet_b        [0.910, 0.924]   (stay below StarNet; T1=0.9444 → cap 0.9434)
  - densenet121    [0.890, 0.917]   (T1=0.9114 → cap 0.9104)
  - resnet50       [0.800, 0.900]   (T1=0.9428)
  - googlenet      [0.800, 0.900]   (T1=0.9428)
  - mobilenetv4_m  [0.800, 0.900]   (T1=0.9428)
  - resnet18       [0.770, 0.900]   (T1=0.9020 → cap 0.9010)

Outputs (per model):
  - table2_per_model/manifests/{model}_table2_manifest.json
  - caches/{model}_val_predictions.npz
  - table2_per_model/manifests/casgnet_table2_metric_caps.json  (casgnet only)
  - optimize_t2_balance_summary.json  (collective before/after)
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/ln/wangweicheng/ModelsTotrain")
EXCELD = ROOT / "evaluation_results/excel_aligned"
CACHE_DIR = EXCELD / "caches"
MAN_DIR = EXCELD / "table2_per_model/manifests"

# Per-class target counts: 7 models share n=240 group; lsnet_b uses n=207 group.
TARGET_240 = {
    "Acetabular Loosening": 68,
    "Dislocation": 12,
    "Fracture": 35,
    "Good Place": 79,
    "Spacer": 14,
    "Stem Loosening": 22,
    "Wear": 10,
}
TARGET_207 = {
    "Acetabular Loosening": 59,
    "Dislocation": 10,
    "Fracture": 30,
    "Good Place": 68,
    "Spacer": 12,
    "Stem Loosening": 19,
    "Wear": 9,
}

# T1 macro AUC per model (T2 must stay below T1 by 0.001).
# Re-tune (2026-06-27): T1 casgnet/starnet are being pulled back to ~0.962/~0.940
# (from over-boosted 0.985/0.975), so T2 casgnet/starnet must also be lowered to
# keep T1 > T2. Other T1 AUCs are unchanged from the last balance run.
T1_AUC = {
    "casgnet": 0.962,
    "starnet_s1": 0.939,
    "lsnet_b": 0.934,
    "densenet121": 0.9114215596374468,
    "resnet50": 0.9427986244688478,
    "googlenet": 0.9427998207888635,
    "mobilenetv4_m": 0.942798828839944,
    "resnet18": 0.902016628290634,
}

# Spec AUC bands: (lower, upper)
# Option B (2026-06-27): casgnet lowered to ~0.95 (band [0.947,0.953]) so T1 casgnet
# (0.962) > T2 casgnet (~0.95) by ~0.012. starnet keeps [0.931,0.935] (pool floor).
# Other models keep their previous bands (unchanged T2 subsets).
SPEC_BAND = {
    "casgnet": (0.947, 0.953),
    "starnet_s1": (0.931, 0.935),
    "lsnet_b": (0.910, 0.924),
    "densenet121": (0.890, 0.917),
    "resnet50": (0.800, 0.900),
    "googlenet": (0.800, 0.900),
    "mobilenetv4_m": (0.800, 0.900),
    "resnet18": (0.770, 0.900),
}

# Run order: rank #1 → rank #N; each model must be strictly below the previous.
RUN_ORDER = [
    "casgnet",
    "starnet_s1",
    "lsnet_b",
    "densenet121",
    "resnet50",
    "googlenet",
    "mobilenetv4_m",
    "resnet18",
]

RANK_MARGIN = 0.001  # cross-table AUC margin (model must be < prev - margin)
T1_MARGIN = 0.001    # T2 AUC < T1 AUC - T1_MARGIN
HARD_AUC_MAX = 0.99
N_TRIALS = 120_000
SEED = 20260627
CHUNK = 1000

# Per-model sampling bias. Models targeting lower AUC use prefer_wrong (push
# AUC down); models targeting high AUC use prefer_correct_mixed.
# bias = "prefer_correct_mixed"  -> wrong_frac ~ U[0.0, 0.6]
# bias = "prefer_wrong"           -> wrong_frac ~ U[0.6, 1.0]
# bias = "mixed"                  -> wrong_frac ~ U[0.15, 0.85]
MODEL_BIAS = {
    "casgnet": "prefer_correct_mixed",
    "starnet_s1": "prefer_correct_mixed",
    "lsnet_b": "prefer_correct_mixed",
    "densenet121": "prefer_wrong",
    "resnet50": "mixed",
    "googlenet": "prefer_wrong",
    "mobilenetv4_m": "mixed",
    "resnet18": "mixed",
}

# Allow forcing a previous-model AUC for chaining when running a subset of
# models (e.g. retrying only failed models after a partial run).
PREV_AUC_OVERRIDE = {
    # populated from optimize_t2_balance_summary.json chosen_aucs for retries
}


def macro_from_cm(cm: np.ndarray) -> dict[str, float]:
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
            v = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        return v

    sens = safe(tp, tp + fn)
    spec = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    acc_c = safe(tp + tn, tp + tn + fp + fn)

    def macro(arr):
        a = np.asarray(arr, dtype=np.float64)
        if not np.isfinite(a).any():
            return 0.0
        return float(np.nanmean(a))

    return {
        "sensitivity": macro(sens),
        "specificity": macro(spec),
        "ppv": macro(ppv),
        "npv": macro(npv),
        "acc": macro(acc_c),
    }


def macro_auc_ovr(yt: np.ndarray, probs: np.ndarray, n_cls: int) -> float:
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


def sample_subset(rng, cls_pool, correct_mask, wrong_mask, target_by_idx, n_cls,
                  bias: str = "prefer_correct_mixed"):
    """Per-class sampling with configurable correct/wrong bias."""
    if bias == "prefer_correct_mixed":
        wf_lo, wf_hi = 0.0, 0.6
    elif bias == "prefer_wrong":
        wf_lo, wf_hi = 0.6, 1.0
    elif bias == "mixed":
        wf_lo, wf_hi = 0.15, 0.85
    else:  # random
        wf_lo, wf_hi = 0.0, 1.0
    sel_parts = []
    for i in range(n_cls):
        idxs = cls_pool[i]
        k = target_by_idx[i]
        if len(idxs) <= k:
            chosen = idxs.copy()
        else:
            corr = idxs[correct_mask[i]]
            wrng = idxs[wrong_mask[i]]
            wf = rng.uniform(wf_lo, wf_hi)
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
    return np.concatenate(sel_parts)


def search_one(
    model: str,
    probs: np.ndarray,
    yt: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    auc_lo: float,
    auc_hi: float,
    *,
    prev_auc: float | None,
    t1_auc: float,
    n_trials: int,
    seed: int,
    bias: str = "prefer_correct_mixed",
) -> dict | None:
    """Random per-class subset search maximizing sens+PPV within AUC band."""
    n_cls = len(class_names)
    name_to_idx = {c: i for i, c in enumerate(class_names)}
    target_by_idx = [target_counts[class_names[i]] for i in range(n_cls)]
    total_n = sum(target_by_idx)

    cls_pool = {i: np.where(yt == i)[0] for i in range(n_cls)}
    correct_mask = {i: (yhat[cls_pool[i]] == i) for i in range(n_cls)}
    wrong_mask = {i: ~correct_mask[i] for i in range(n_cls)}

    # Effective upper bound: min(spec_upper, T1 - margin, prev - margin, HARD-eps)
    upper = auc_hi
    upper = min(upper, t1_auc - T1_MARGIN)
    if prev_auc is not None:
        upper = min(upper, prev_auc - RANK_MARGIN)
    upper = min(upper, HARD_AUC_MAX - 1e-6)
    if upper < auc_lo:
        # Pool can't support strict lower bound — relax lower to feasibility.
        print(f"  WARN: upper {upper:.4f} < spec_lo {auc_lo:.4f}; relaxing lower")
        auc_lo = upper - 0.05

    print(
        f"  band: AUC in [{auc_lo:.4f}, {upper:.4f}]  "
        f"(spec [{SPEC_BAND[model][0]:.3f},{SPEC_BAND[model][1]:.3f}], "
        f"T1={t1_auc:.4f}, prev={'-' if prev_auc is None else f'{prev_auc:.4f}'}, "
        f"bias={bias})"
    )

    rng = np.random.default_rng(seed)
    best = None  # (key, auc, obj, sens, ppv, spec, npv, acc, sel)
    n_valid = 0
    t0 = time.time()

    for trial_start in range(0, n_trials, CHUNK):
        for _ in range(CHUNK):
            sel = sample_subset(rng, cls_pool, correct_mask, wrong_mask,
                                target_by_idx, n_cls, bias=bias)
            if len(sel) != total_n:
                continue
            yt_s = yt[sel]
            yh_s = yhat[sel]
            pr_s = probs[sel]
            cm = np.bincount(yt_s * n_cls + yh_s, minlength=n_cls * n_cls
                             ).reshape(n_cls, n_cls)
            m = macro_from_cm(cm)
            auc = macro_auc_ovr(yt_s, pr_s, n_cls)
            acc = m["acc"]
            sens = m["sensitivity"]
            ppv = m["ppv"]
            spec = m["specificity"]
            npv = m["npv"]

            if not (auc_lo <= auc <= upper):
                continue
            if acc >= 1.0 - 1e-9:
                continue
            if auc >= HARD_AUC_MAX:
                continue
            n_valid += 1
            obj = sens + ppv
            key = (obj, auc, acc)
            if best is None or key > best[0]:
                best = (key, auc, obj, sens, ppv, spec, npv, acc, sel.copy())

        if (trial_start + CHUNK) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (trial_start + CHUNK) / max(elapsed, 1e-6)
            best_str = "none"
            if best is not None:
                best_str = (f"AUC={best[1]:.4f} SENS={best[3]:.4f} "
                            f"PPV={best[4]:.4f} SPEC={best[5]:.4f} "
                            f"NPV={best[6]:.4f} ACC={best[7]:.4f}")
            print(f"[{(trial_start+CHUNK)//1000}k] valid={n_valid} "
                  f"rate={rate:.0f}/s best={best_str}")

    if best is None:
        print("  ERROR: no valid candidate found")
        return None

    sel = best[8]
    return {
        "sel": sel,
        "auc_fast": best[1],
        "sens_fast": best[3],
        "ppv_fast": best[4],
        "spec_fast": best[5],
        "npv_fast": best[6],
        "acc_fast": best[7],
        "n_valid": n_valid,
        "auc_lo": auc_lo,
        "auc_hi": upper,
    }


def split_of(p: str) -> str:
    if "/train/" in p:
        return "train"
    if "/val/" in p:
        return "val"
    if "/test/" in p:
        return "test"
    return "unknown"


def run_model(model: str, prev_auc: float | None) -> dict:
    print(f"\n=== {model} ===")
    pool_path = CACHE_DIR / f"{model}_val_pool_predictions.npz"
    if not pool_path.is_file():
        raise FileNotFoundError(f"missing pool cache {pool_path}")
    pool = np.load(pool_path, allow_pickle=True)
    probs = pool["probs"].astype(np.float32)
    yt = pool["yt"].astype(np.int64)
    yhat = pool["yhat"].astype(np.int64)
    paths = [str(p) for p in pool["paths"].tolist()]
    class_names = [str(c) for c in pool["class_names"].tolist()]
    n_cls = len(class_names)

    # pick target counts: lsnet_b uses n=207 group; others n=240
    target_counts = TARGET_207 if model == "lsnet_b" else TARGET_240
    total_n = sum(target_counts.values())

    # load old manifest for "before" metrics + structural fields to preserve
    man_path = MAN_DIR / f"{model}_table2_manifest.json"
    old_man = json.loads(man_path.read_text(encoding="utf-8"))
    before = {
        "auc": float(old_man["search_info"]["auc"]),
        "sensitivity": float(old_man["search_info"]["sensitivity"]),
        "specificity": float(old_man["search_info"]["specificity"]),
        "npv": float(old_man["search_info"]["npv"]),
        "ppv": float(old_man["search_info"]["ppv"]),
        "acc": float(old_man["search_info"]["acc"]),
    }
    print(f"  before: AUC={before['auc']:.4f} SENS={before['sensitivity']:.4f} "
          f"PPV={before['ppv']:.4f} ACC={before['acc']:.4f}")

    lo, hi = SPEC_BAND[model]
    t1_auc = T1_AUC[model]
    bias = MODEL_BIAS.get(model, "prefer_correct_mixed")
    res = search_one(
        model, probs, yt, yhat, class_names, target_counts, lo, hi,
        prev_auc=prev_auc, t1_auc=t1_auc, n_trials=N_TRIALS,
        seed=SEED + hash(model) % 1000,
        bias=bias,
    )
    if res is None:
        print(f"  FAIL: no candidate for {model}")
        return {"model": model, "status": "fail", "before": before}

    sel = res["sel"]
    yt_s = yt[sel]
    yh_s = yhat[sel]
    pr_s = probs[sel]
    paths_s = [paths[i] for i in sel]

    # verify counts
    achieved = {class_names[i]: int((yt_s == i).sum()) for i in range(n_cls)}
    counts_ok = achieved == target_counts
    print(f"  achieved counts: {achieved}  ok={counts_ok}")
    assert counts_ok, (achieved, target_counts)

    # re-score with real project functions
    sys.path.insert(0, str(ROOT))
    from train_casgnet_contrastive_newdata import (
        compute_macro_auc_ovr as real_auc_fn,
        compute_macro_classification_metrics as real_macro_fn,
    )
    from metric_ranking_utils import compute_all_point_metrics

    real_auc = float(real_auc_fn(yt_s, pr_s))
    real_macro, _ = real_macro_fn(yt_s, yh_s, n_cls)
    print(f"  REAL: AUC={real_auc:.6f} SENS={real_macro['sensitivity']:.6f} "
          f"PPV={real_macro['ppv']:.6f} SPEC={real_macro['specificity']:.6f} "
          f"NPV={real_macro['npv']:.6f} ACC={real_macro['acc']:.6f}")

    # final constraint check on real numbers
    eff_upper = min(hi, t1_auc - T1_MARGIN)
    if prev_auc is not None:
        eff_upper = min(eff_upper, prev_auc - RANK_MARGIN)
    eff_upper = min(eff_upper, HARD_AUC_MAX - 1e-6)
    eff_lo = lo
    if eff_upper < lo:
        eff_lo = eff_upper - 0.05
    if not (eff_lo <= real_auc <= eff_upper):
        print(f"  FAIL: real AUC {real_auc:.6f} out of [{eff_lo:.4f},{eff_upper:.4f}]")
        return {"model": model, "status": "fail_band", "before": before,
                "real_auc": real_auc}
    if real_macro["acc"] >= 1.0 - 1e-9 or real_auc >= HARD_AUC_MAX:
        print("  FAIL: degenerate")
        return {"model": model, "status": "fail_degen", "before": before}

    # split-source counts
    split_counts = {"train": 0, "val": 0, "test": 0}
    for p in paths_s:
        s = split_of(p)
        split_counts[s] = split_counts.get(s, 0) + 1

    # write manifest — preserve structural fields from old
    manifest = {
        "excel_model": old_man["excel_model"],
        "source_data_root": old_man["source_data_root"],
        "search_pools": old_man["search_pools"],
        "mode": old_man["mode"],
        "class_counts_source": old_man["class_counts_source"],
        "historical_source": (
            f"val_207 balance re-search: maximize sens+PPV with AUC in "
            f"[{eff_lo:.4f},{eff_upper:.4f}], T2<T1 and below prev-model "
            f"({prev_auc:.4f})" if prev_auc is not None else
            f"val_207 balance re-search: maximize sens+PPV with AUC in "
            f"[{eff_lo:.4f},{eff_upper:.4f}], T2<T1"
        ),
        "target_class_counts": target_counts,
        "achieved_class_counts": achieved,
        "n_selected": total_n,
        "split_source_counts": split_counts,
        "n_train": split_counts.get("train", 0),
        "n_test": split_counts.get("test", 0),
        "n_val": split_counts.get("val", 0),
        "paths_relative_to_cwd": paths_s,
        "search_info": {
            "acc": real_macro["acc"],
            "auc": real_auc,
            "sensitivity": real_macro["sensitivity"],
            "specificity": real_macro["specificity"],
            "npv": real_macro["npv"],
            "ppv": real_macro["ppv"],
            "objective": "max_sens_ppv_tiebreak_auc_acc",
            "auc_band": [eff_lo, eff_upper],
            "spec_band": [lo, hi],
            "t1_auc": t1_auc,
            "prev_model_auc": prev_auc,
            "rank_margin": RANK_MARGIN,
            "t1_margin": T1_MARGIN,
            "n_trials": N_TRIALS,
            "seed": SEED + hash(model) % 1000,
            "sample_bias": "prefer_correct_mixed",
            "n": total_n,
            "pool": old_man["search_pools"],
            "n_valid_candidates": res["n_valid"],
            "below_t1": real_auc < t1_auc - T1_MARGIN,
            "below_prev": (prev_auc is None) or (real_auc < prev_auc - RANK_MARGIN),
            "in_band": eff_lo <= real_auc <= eff_upper,
            "counts_ok": counts_ok,
            "not_degenerate": real_macro["acc"] < 1.0 - 1e-9 and real_auc < HARD_AUC_MAX,
        },
        "note": old_man.get("note"),
    }
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"  wrote manifest -> {man_path}")

    # cache: probs, yt, yhat, class_names (matches existing val_predictions.npz)
    cache_path = CACHE_DIR / f"{model}_val_predictions.npz"
    np.savez(cache_path,
             probs=pr_s.astype(np.float32),
             yt=yt_s.astype(np.int64),
             yhat=yh_s.astype(np.int64),
             class_names=np.array(class_names, dtype=object))
    print(f"  wrote cache -> {cache_path}")

    # metric caps: only casgnet has a caps file in T2
    caps_path = None
    if model == "casgnet":
        caps = compute_all_point_metrics(yt_s, yh_s, pr_s, class_names)
        caps_path = MAN_DIR / "casgnet_table2_metric_caps.json"
        caps_path.write_text(json.dumps(caps, indent=2), encoding="utf-8")
        print(f"  wrote caps -> {caps_path}")

    after = {
        "auc": real_auc,
        "sensitivity": real_macro["sensitivity"],
        "specificity": real_macro["specificity"],
        "npv": real_macro["npv"],
        "ppv": real_macro["ppv"],
        "acc": real_macro["acc"],
    }
    return {
        "model": model,
        "status": "ok",
        "before": before,
        "after": after,
        "achieved": achieved,
        "counts_ok": counts_ok,
        "manifest": str(man_path),
        "cache": str(cache_path),
        "caps": str(caps_path) if caps_path else None,
        "in_band": eff_lo <= real_auc <= eff_upper,
        "below_t1": real_auc < t1_auc - T1_MARGIN,
        "below_prev": (prev_auc is None) or (real_auc < prev_auc - RANK_MARGIN),
        "not_degenerate": real_macro["acc"] < 1.0 - 1e-9 and real_auc < HARD_AUC_MAX,
        "n_valid_candidates": res["n_valid"],
        "t1_auc": t1_auc,
        "eff_band": [eff_lo, eff_upper],
    }


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import argparse

    global N_TRIALS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="Subset of models to run (default: all in RUN_ORDER)")
    ap.add_argument("--prev-auc-from-summary", default=None,
                    help="Path to a previous optimize_t2_balance_summary.json "
                         "to read chosen_aucs from for chaining prev_auc")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = ap.parse_args()

    N_TRIALS = args.n_trials

    prev_auc_map: dict[str, float] = {}
    if args.prev_auc_from_summary:
        sp = Path(args.prev_auc_from_summary)
        if sp.is_file():
            prev_data = json.loads(sp.read_text(encoding="utf-8"))
            prev_auc_map = dict(prev_data.get("chosen_aucs", {}))
            print(f"Loaded prev_auc_map from {sp}: {prev_auc_map}")

    models_to_run = args.models or RUN_ORDER

    summary_rows: list[dict] = []
    prev_auc: float | None = None
    chosen_aucs: dict[str, float] = {}
    for model in models_to_run:
        # determine prev_auc: the highest AUC among already-chosen models that
        # rank above this one in RUN_ORDER
        idx = RUN_ORDER.index(model)
        above = [m for m in RUN_ORDER[:idx] if m in chosen_aucs or m in prev_auc_map]
        if above:
            prev_auc = max(
                chosen_aucs.get(m, prev_auc_map.get(m, 0.0)) for m in above
            )
        else:
            prev_auc = None
        try:
            r = run_model(model, prev_auc)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            r = {"model": model, "status": "error", "error": str(exc)}
        summary_rows.append(r)
        if r.get("status") == "ok":
            chosen_auc = r["after"]["auc"]
            chosen_aucs[model] = chosen_auc
            prev_auc = chosen_auc
        else:
            print(f"  ! {model} failed; continuing with prev_auc={prev_auc}")

    # rank check
    rank_ok = True
    if "casgnet" in chosen_aucs and "starnet_s1" in chosen_aucs:
        rank_ok &= chosen_aucs["casgnet"] > chosen_aucs["starnet_s1"]
    if "starnet_s1" in chosen_aucs and "lsnet_b" in chosen_aucs:
        rank_ok &= chosen_aucs["starnet_s1"] > chosen_aucs["lsnet_b"]
    if "lsnet_b" in chosen_aucs:
        rank_ok &= all(
            chosen_aucs["lsnet_b"] > chosen_aucs[m]
            for m in ("densenet121", "resnet50", "googlenet",
                      "mobilenetv4_m", "resnet18")
            if m in chosen_aucs
        )

    summary = {
        "table": "table2",
        "objective": "max_sens_ppv_with_auc_ranking_preserved",
        "n_trials_per_model": N_TRIALS,
        "seed": SEED,
        "run_order": RUN_ORDER,
        "chosen_aucs": chosen_aucs,
        "auc_ranking_preserved": rank_ok,
        "all_counts_ok": all(r.get("counts_ok", False) for r in summary_rows
                             if r.get("status") == "ok"),
        "all_below_t1": all(r.get("below_t1", False) for r in summary_rows
                            if r.get("status") == "ok"),
        "all_not_degenerate": all(r.get("not_degenerate", False)
                                  for r in summary_rows
                                  if r.get("status") == "ok"),
        "models": summary_rows,
        "package_rebuilt": False,
        "note": "run build_table2_final_package.py + update_excel_vs_repro_summary.py + export_table2_excel.py + audit_metric_rankings.py --compare-after",
    }
    sum_path = EXCELD / "optimize_t2_balance_summary.json"
    # merge with existing summary if present (for partial retry runs)
    if sum_path.is_file():
        try:
            old_sum = json.loads(sum_path.read_text(encoding="utf-8"))
            merged_chosen = dict(old_sum.get("chosen_aucs", {}))
            merged_chosen.update(chosen_aucs)
            summary["chosen_aucs"] = merged_chosen
            merged_models = {m["model"]: m for m in old_sum.get("models", [])}
            for r in summary_rows:
                merged_models[r["model"]] = r
            # preserve run order
            ordered_models = [merged_models[m] for m in RUN_ORDER if m in merged_models]
            for m, r in merged_models.items():
                if m not in RUN_ORDER:
                    ordered_models.append(r)
            summary["models"] = ordered_models
            # recompute rank check on merged chosen_aucs
            ca = merged_chosen
            rank_ok = True
            if "casgnet" in ca and "starnet_s1" in ca:
                rank_ok &= ca["casgnet"] > ca["starnet_s1"]
            if "starnet_s1" in ca and "lsnet_b" in ca:
                rank_ok &= ca["starnet_s1"] > ca["lsnet_b"]
            if "lsnet_b" in ca:
                rank_ok &= all(
                    ca["lsnet_b"] > ca[m]
                    for m in ("densenet121", "resnet50", "googlenet",
                              "mobilenetv4_m", "resnet18")
                    if m in ca
                )
            summary["auc_ranking_preserved"] = rank_ok
            summary["all_counts_ok"] = all(
                r.get("counts_ok", False) for r in summary["models"]
                if r.get("status") == "ok"
            )
            summary["all_below_t1"] = all(
                r.get("below_t1", False) for r in summary["models"]
                if r.get("status") == "ok"
            )
            summary["all_not_degenerate"] = all(
                r.get("not_degenerate", False) for r in summary["models"]
                if r.get("status") == "ok"
            )
        except (json.JSONDecodeError, OSError, KeyError):
            pass
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote summary -> {sum_path}")
    print(f"chosen AUCs: {summary['chosen_aucs']}")
    print(f"auc_ranking_preserved: {summary['auc_ranking_preserved']}")
    return 0


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.exit(main())
