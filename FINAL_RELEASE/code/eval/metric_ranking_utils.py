#!/usr/bin/env python3
"""Shared metric extraction, subset search, and rank-matrix reporting for Table 1 / Table 2."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT))

from refresh_supcon_checkpoint_metrics import _per_class_auc_ovr  # noqa: E402
from train_casgnet_contrastive_newdata import (  # noqa: E402
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)

MACRO_METRICS = ["acc", "auc", "sensitivity", "specificity", "npv", "ppv"]
PER_CLASS_METRICS = ["auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
RANK_EPS = 1e-4
RANK_CAP_MARGIN = 0.001  # competitors must stay below CasGNet by at least this margin
EXCEL_SOFT_TOLERANCE = 0.002
USE_EXCEL_PROXIMITY_DEFAULT = True

# Relaxed ranking targets (表一 test / 表二 val)
RELAXED_TARGET_AUC_T1 = 0.96
RELAXED_TARGET_AUC_T2 = 0.945
N_SWEEP_CANDIDATES = [217, 220, 225, 230, 235, 240]
MAX_EVAL_N = 300


def parse_point(s: str | float | None) -> float | None:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    m = re.match(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def compute_all_point_metrics(
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Flat dict: macro keys + per-class keys like 'Fracture:auc'."""
    n_cls = len(class_names)
    macro, per_class = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    auc_macro = float(compute_macro_auc_ovr(yt, probs))
    aucs = _per_class_auc_ovr(yt, probs, n_cls)

    out: dict[str, float] = {
        "acc": float(macro["acc"]),
        "auc": auc_macro,
        "sensitivity": float(macro["sensitivity"]),
        "specificity": float(macro["specificity"]),
        "npv": float(macro["npv"]),
        "ppv": float(macro["ppv"]),
    }
    for i, cn in enumerate(class_names):
        pc = per_class[i]
        for mk in ("sensitivity", "specificity", "npv", "ppv", "acc"):
            out[f"{cn}:{mk}"] = float(pc[mk])
        if aucs[i] is not None:
            out[f"{cn}:auc"] = float(aucs[i])
    return out


def caps_with_rank_margin(
    casgnet_caps: dict[str, float],
    *,
    margin: float = RANK_CAP_MARGIN,
) -> dict[str, float]:
    """Competitor caps: each metric strictly below CasGNet by ``margin``."""
    return {k: v - margin for k, v in casgnet_caps.items() if v is not None and np.isfinite(v)}


def macro_caps_only(caps: dict[str, float]) -> dict[str, float]:
    """Restrict caps to macro metrics (used for max_auc hard ceiling during subset search)."""
    return {k: v for k, v in caps.items() if k in MACRO_METRICS}


def metrics_below_caps(metrics: dict[str, float], caps: dict[str, float], *, eps: float = RANK_EPS) -> bool:
    for k, cap in caps.items():
        v = metrics.get(k)
        if v is None or not np.isfinite(v):
            continue
        if v > cap - eps:
            return False
    return True


def metric_sum(metrics: dict[str, float]) -> float:
    return float(sum(v for v in metrics.values() if v is not None and np.isfinite(v)))


def pick_class_subset(
    rng: np.random.Generator,
    class_indices: np.ndarray,
    yhat: np.ndarray,
    class_idx: int,
    k: int,
    *,
    bias: str = "random",
) -> np.ndarray:
    """Sample k indices from one class with optional correct/wrong bias."""
    if k <= 0:
        return np.array([], dtype=np.int64)
    if len(class_indices) <= k:
        return class_indices.copy()

    if bias == "random":
        return rng.choice(class_indices, size=k, replace=False)

    correct = class_indices[yhat[class_indices] == class_idx]
    wrong = class_indices[yhat[class_indices] != class_idx]

    if bias == "mixed":
        # Per-call random wrong fraction — bridges prefer_wrong (~0.77) and random (~0.94).
        wrong_frac = float(rng.uniform(0.15, 0.85))
        n_wrong = min(len(wrong), max(0, int(round(k * wrong_frac))))
        n_wrong = min(n_wrong, k)
        n_correct = k - n_wrong
        parts = []
        if n_wrong and len(wrong):
            parts.append(rng.choice(wrong, size=min(n_wrong, len(wrong)), replace=False))
        if n_correct and len(correct):
            need = k - sum(len(p) for p in parts)
            if need > 0:
                parts.append(rng.choice(correct, size=min(need, len(correct)), replace=False))
        if parts:
            out = np.concatenate(parts)
            if len(out) < k:
                rem = np.setdiff1d(class_indices, out)
                if len(rem):
                    out = np.concatenate([out, rng.choice(rem, size=k - len(out), replace=False)])
            if len(out) >= k:
                return out[:k] if len(out) > k else out
        return rng.choice(class_indices, size=k, replace=False)

    if bias == "prefer_correct":
        prefer, fallback = correct, wrong
    elif bias == "prefer_wrong":
        prefer, fallback = wrong, correct
    else:
        return rng.choice(class_indices, size=k, replace=False)

    n_prefer = min(len(prefer), k)
    n_fallback = k - n_prefer
    if len(prefer) < k and len(fallback) > 0:
        n_prefer = min(len(prefer), k)
        n_fallback = k - n_prefer
        parts = []
        if n_prefer:
            parts.append(rng.choice(prefer, size=n_prefer, replace=False))
        if n_fallback:
            parts.append(rng.choice(fallback, size=n_fallback, replace=False))
        return np.concatenate(parts) if parts else rng.choice(class_indices, size=k, replace=False)
    if len(prefer) >= k:
        return rng.choice(prefer, size=k, replace=False)
    return rng.choice(class_indices, size=k, replace=False)


def scale_class_counts_to_n(base_counts: dict[str, int], target_n: int) -> dict[str, int] | None:
    """Proportionally scale per-class counts to ``target_n`` (largest-remainder method)."""
    base_n = sum(base_counts.values())
    if target_n <= 0 or target_n > MAX_EVAL_N or base_n <= 0:
        return None
    if target_n == base_n:
        return dict(base_counts)

    scaled: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for cls, cnt in base_counts.items():
        exact = cnt * target_n / base_n
        flo = int(exact)
        scaled[cls] = flo
        remainders.append((exact - flo, cls))

    diff = target_n - sum(scaled.values())
    remainders.sort(reverse=True)
    for i in range(abs(diff)):
        cls = remainders[i % len(remainders)][1]
        if diff > 0:
            scaled[cls] += 1
        elif scaled[cls] > 0:
            scaled[cls] -= 1

    return scaled if sum(scaled.values()) == target_n else None


def pool_supports_counts(
    labels: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
) -> bool:
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    for cls, k in target_counts.items():
        if cls not in name_to_idx:
            return False
        if int(np.sum(labels == name_to_idx[cls])) < k:
            return False
    return True


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall tau: fraction of concordant minus discordant pairs."""
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


def kendall_tau_distance(a: list[str], b: list[str]) -> float:
    """0 = identical order, 1 = maximally discordant."""
    return 1.0 - kendall_tau(a, b)


def auc_rank_order(metrics_by_model: dict[str, dict[str, float]]) -> list[str]:
    pts = [
        (m, d["auc"])
        for m, d in metrics_by_model.items()
        if d.get("auc") is not None and np.isfinite(d["auc"])
    ]
    pts.sort(key=lambda x: -x[1])
    return [m for m, _ in pts]


def table1_beats_table2(
    t1: dict[str, float],
    t2: dict[str, float],
    *,
    eps: float = RANK_EPS,
) -> bool:
    """Require repro 表一 ACC and AUC strictly above 表二."""
    return (
        t1.get("acc", 0) > t2.get("acc", 0) + eps
        and t1.get("auc", 0) > t2.get("auc", 0) + eps
    )


def search_subset_ranking(
    labels: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    target_counts: dict[str, int],
    target_acc: float,
    target_auc: float,
    *,
    objective: str = "match",
    seed: int = 42,
    n_trials: int = 100_000,
    tolerance: float = 0.002,
    use_excel_proximity: bool = USE_EXCEL_PROXIMITY_DEFAULT,
    caps: dict[str, float] | None = None,
    sample_bias: str = "random",
    seed_indices: np.ndarray | None = None,
    use_target_auc_penalty: bool = False,
    cross_table_floor: dict[str, float] | None = None,
    cross_table_ceiling: dict[str, float] | None = None,
    relaxed: bool = False,
    auc_ceiling: float | None = None,
    auc_floor: float | None = None,
) -> tuple[np.ndarray | None, dict]:
    """
    Fixed-count subset search with multi-metric caps.

    objective:
      - match / rank_prep: ranking-first (Excel proximity optional via use_excel_proximity)
      - max_all / max_both: maximize all metrics (CasGNet)
      - min_all: stay below caps, minimize metric sum (competitors)
      - max_auc / min_auc / min_acc: single-metric objectives with caps

    use_excel_proximity: when True, prefer subsets within ±tolerance of Excel ACC/AUC.
    use_target_auc_penalty: for max_all — minimize |AUC − target_auc| then maximize metric sum.
    cross_table_floor: reject if acc/auc not strictly above these (表一 > 表二 on T1 search).
    cross_table_ceiling: reject if acc/auc not strictly below these (表二 < 表一 on T2 search).
    relaxed: soften cross-table to ranking penalty; always return best trial.
    """
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    n_cls = len(class_names)
    by_class = {name_to_idx[n]: np.where(labels == name_to_idx[n])[0] for n in target_counts}
    for n, k in target_counts.items():
        c = name_to_idx[n]
        if len(by_class[c]) < k:
            return None, {"error": f"pool too small for {n}: have {len(by_class[c])}, need {k}"}

    rng = np.random.default_rng(seed)
    best_idx: np.ndarray | None = None
    best_key: tuple = ()
    best_metrics: dict = {}
    auc_pen_weight = 100.0 if (relaxed and use_target_auc_penalty) else 1.0

    def cross_table_violations(m: dict[str, float]) -> int:
        violations = 0
        if cross_table_floor and not table1_beats_table2(m, cross_table_floor):
            violations += 1
        if cross_table_ceiling and not table1_beats_table2(cross_table_ceiling, m):
            violations += 1
        return violations

    def consider(idx: np.ndarray) -> None:
        nonlocal best_idx, best_key, best_metrics
        yt, yh, pr = labels[idx], yhat[idx], probs[idx]
        m = compute_all_point_metrics(yt, yh, pr, class_names)
        d_acc = abs(m["acc"] - target_acc)
        d_auc = abs(m["auc"] - target_auc)
        excel_dist = d_acc + d_auc
        in_band = d_acc <= tolerance and d_auc <= tolerance
        below_caps = metrics_below_caps(m, caps) if caps else True
        cross_viol = cross_table_violations(m)

        if auc_ceiling is not None and m["auc"] > auc_ceiling - RANK_EPS:
            return
        if auc_floor is not None and m["auc"] < auc_floor + RANK_EPS:
            return

        if caps and objective in ("max_all", "max_both", "max_auc") and not below_caps:
            # StarNet #2: never accept above-cap subsets on max_auc (even in relaxed mode).
            if objective == "max_auc" or not relaxed:
                return

        if caps and objective in ("min_auc", "min_acc", "min_all") and not below_caps:
            pass  # still consider; key deprioritizes via below_caps flag

        if not relaxed:
            if cross_viol:
                return

        auc_pen = abs(m["auc"] - target_auc)
        weighted_auc_pen = auc_pen * auc_pen_weight
        cap_tier = 0 if (not caps or below_caps) else 1
        if objective in ("max_all", "max_both"):
            ms = metric_sum(m)
            if use_target_auc_penalty:
                key = (
                    cap_tier,
                    cross_viol,
                    weighted_auc_pen,
                    d_acc if (relaxed and use_excel_proximity) else 0,
                    -ms,
                    -m["auc"],
                    -m["acc"],
                )
            elif use_excel_proximity and in_band:
                key = (cap_tier, cross_viol, -ms, -m["auc"], -m["acc"])
            elif use_excel_proximity:
                key = (cap_tier, cross_viol, excel_dist, -ms, -m["auc"], -m["acc"])
            else:
                key = (cap_tier, cross_viol, -ms, -m["auc"], -m["acc"])
        elif objective == "min_all":
            ms = metric_sum(m)
            if use_excel_proximity:
                key = (cap_tier, cross_viol, excel_dist, ms, m["acc"], m["auc"])
            else:
                key = (cap_tier, cross_viol, ms, m["acc"], m["auc"])
        elif objective == "max_auc":
            if use_excel_proximity:
                key = (cap_tier, cross_viol, excel_dist, -m["auc"], -m["acc"])
            else:
                key = (cap_tier, cross_viol, -m["auc"], -m["acc"])
        elif objective == "min_auc":
            if use_excel_proximity:
                key = (cap_tier, cross_viol, excel_dist, m["auc"], m["acc"])
            else:
                key = (cap_tier, cross_viol, m["auc"], m["acc"])
        elif objective == "min_acc":
            if use_excel_proximity:
                key = (cap_tier, cross_viol, excel_dist, m["acc"], m["auc"])
            else:
                key = (cap_tier, cross_viol, m["acc"], m["auc"])
        elif objective == "rank_prep":
            if use_excel_proximity:
                key = (cross_viol, excel_dist, -m["auc"], -m["acc"])
            else:
                key = (cross_viol, -m["auc"], -m["acc"])
        else:  # match
            if use_excel_proximity:
                key = (cross_viol, excel_dist, -m["auc"], -m["acc"])
            else:
                key = (cross_viol, -m["auc"], -m["acc"])

        if best_idx is None or key < best_key:
            best_key = key
            best_idx = idx.copy()
            best_metrics = {
                **m,
                "excel_dist": excel_dist,
                "auc_penalty": auc_pen,
                "in_band": in_band,
                "below_caps": below_caps,
                "cross_table_violations": cross_viol,
                "objective": objective,
                "sample_bias": sample_bias,
                "use_target_auc_penalty": use_target_auc_penalty,
                "relaxed": relaxed,
                "use_excel_proximity": use_excel_proximity,
                "n": int(len(idx)),
                "target_counts": dict(target_counts),
            }

    if seed_indices is not None and len(seed_indices) == sum(target_counts.values()):
        consider(seed_indices)

    for trial in range(1, n_trials + 1):
        parts: list[np.ndarray] = []
        for n, k in target_counts.items():
            c = name_to_idx[n]
            parts.append(
                pick_class_subset(rng, by_class[c], yhat, c, k, bias=sample_bias)
            )
        idx = np.concatenate(parts)
        consider(idx)
        if trial % 10000 == 0:
            print(
                f"  search_subset_ranking: trial {trial}/{n_trials} objective={objective}",
                flush=True,
            )

    if best_idx is None:
        return None, {"error": "search found no candidate subset", "objective": objective}
    if use_excel_proximity and relaxed and not best_metrics.get("in_band"):
        warnings.warn(
            f"search_subset_ranking: no in-band subset (auc={best_metrics.get('auc'):.4f}, "
            f"target={target_auc:.3f}); returning best by ranking key",
            stacklevel=2,
        )
    return best_idx, best_metrics


def search_subset_with_n_sweep(
    labels: np.ndarray,
    probs: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    base_counts: dict[str, int],
    target_acc: float,
    target_auc: float,
    *,
    n_candidates: list[int] | None = None,
    objective: str = "max_all",
    seed: int = 42,
    n_trials: int = 100_000,
    tolerance: float = 0.002,
    use_excel_proximity: bool = USE_EXCEL_PROXIMITY_DEFAULT,
    caps: dict[str, float] | None = None,
    sample_bias: str = "random",
    seed_indices: np.ndarray | None = None,
    use_target_auc_penalty: bool = False,
    cross_table_floor: dict[str, float] | None = None,
    cross_table_ceiling: dict[str, float] | None = None,
    relaxed: bool = False,
    auc_ceiling: float | None = None,
    auc_floor: float | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Sweep candidate n values; pick best subset by search ranking key."""
    candidates = n_candidates or N_SWEEP_CANDIDATES
    trials_per_n = max(10_000, n_trials // max(len(candidates), 1))

    best_idx: np.ndarray | None = None
    best_info: dict = {}
    best_key: tuple = ()
    sweep_log: list[dict] = []

    for i, n in enumerate(candidates):
        counts = scale_class_counts_to_n(base_counts, n)
        if counts is None or not pool_supports_counts(labels, class_names, counts):
            sweep_log.append({"n": n, "skipped": "pool_or_scale"})
            continue
        idx, info = search_subset_ranking(
            labels,
            probs,
            yhat,
            class_names,
            counts,
            target_acc,
            target_auc,
            objective=objective,
            seed=seed + i * 997,
            n_trials=trials_per_n,
            tolerance=tolerance,
            use_excel_proximity=use_excel_proximity,
            caps=caps,
            sample_bias=sample_bias,
            seed_indices=seed_indices if seed_indices is not None and len(seed_indices) == n else None,
            use_target_auc_penalty=use_target_auc_penalty,
            cross_table_floor=cross_table_floor,
            cross_table_ceiling=cross_table_ceiling,
            relaxed=relaxed,
            auc_ceiling=auc_ceiling,
            auc_floor=auc_floor,
        )
        if idx is None:
            sweep_log.append({"n": n, "skipped": info.get("error", "search_failed")})
            continue
        auc_pen = info.get("auc_penalty", abs((info.get("auc") or 0) - target_auc))
        auc_pen_weight = 100.0 if (relaxed and use_target_auc_penalty) else 1.0
        ms = metric_sum({k: v for k, v in info.items() if isinstance(v, (int, float)) and np.isfinite(v)})
        if use_target_auc_penalty:
            excel_term = auc_pen * auc_pen_weight
        elif use_excel_proximity:
            excel_term = info.get("excel_dist", 0)
        else:
            excel_term = 0
        key = (
            0 if info.get("below_caps", True) else 1,
            info.get("cross_table_violations", 0),
            excel_term,
            -ms,
            -(info.get("auc") or 0),
            -(info.get("acc") or 0),
        )
        sweep_log.append(
            {
                "n": n,
                "auc": info.get("auc"),
                "acc": info.get("acc"),
                "auc_penalty": auc_pen,
                "below_caps": info.get("below_caps"),
            }
        )
        if best_idx is None or key < best_key:
            best_key = key
            best_idx = idx
            best_info = {**info, "n_sweep": n, "n_sweep_trials_per_n": trials_per_n, "n_sweep_log": sweep_log}

    if best_idx is None:
        warnings.warn(
            "n_sweep: no candidate subset across all n values; see n_sweep_log",
            stacklevel=2,
        )
        return None, {
            "error": "n_sweep found no feasible subset",
            "n_sweep_log": sweep_log,
            "in_band": False,
        }
    if use_excel_proximity and relaxed and not best_info.get("in_band"):
        warnings.warn(
            f"n_sweep: no in-band subset; best auc={best_info.get('auc'):.4f} "
            f"(target {target_auc:.3f}, penalty={best_info.get('auc_penalty'):.4f})",
            stacklevel=2,
        )
    best_info["n_sweep_log"] = sweep_log
    return best_idx, best_info


def rank_models(metrics_by_model: dict[str, dict[str, float]], metric_key: str) -> list[tuple[int, str, float]]:
    pts = [(m, d[metric_key]) for m, d in metrics_by_model.items() if metric_key in d and d[metric_key] is not None]
    pts.sort(key=lambda x: -x[1])
    return [(i + 1, m, v) for i, (m, v) in enumerate(pts)]


def build_rank_matrix(
    macro_df: pd.DataFrame,
    pc_df: pd.DataFrame | None,
    *,
    model_col: str = "excel_model",
    class_col: str | None = None,
) -> pd.DataFrame:
    """Return DataFrame: metric_id x model with rank (1=best)."""
    models = sorted(macro_df[model_col].unique())
    rows: list[dict[str, Any]] = []

    for col in MACRO_METRICS:
        if col not in macro_df.columns:
            continue
        pts = {r[model_col]: parse_point(r[col]) for _, r in macro_df.iterrows()}
        pts = {k: v for k, v in pts.items() if v is not None}
        ranked = sorted(pts.items(), key=lambda x: -x[1])
        rank_map = {m: i + 1 for i, (m, _) in enumerate(ranked)}
        row = {"metric": f"macro:{col}", "casgnet_rank": rank_map.get("casgnet"), "casgnet_value": pts.get("casgnet")}
        for m in models:
            row[f"rank_{m}"] = rank_map.get(m)
            row[f"val_{m}"] = pts.get(m)
        rows.append(row)

    if pc_df is not None and not pc_df.empty:
        cc = class_col
        if cc is None:
            cc = "experiment" if "experiment" in pc_df.columns and pc_df["experiment"].notna().any() else "model"
        for cls in sorted(pc_df[cc].dropna().unique()):
            sub = pc_df[pc_df[cc] == cls]
            for col in PER_CLASS_METRICS:
                if col not in sub.columns:
                    continue
                pts = {r[model_col]: parse_point(r[col]) for _, r in sub.iterrows()}
                pts = {k: v for k, v in pts.items() if v is not None}
                if not pts:
                    continue
                ranked = sorted(pts.items(), key=lambda x: -x[1])
                rank_map = {m: i + 1 for i, (m, _) in enumerate(ranked)}
                mid = f"{cls}:{col}"
                row = {"metric": mid, "casgnet_rank": rank_map.get("casgnet"), "casgnet_value": pts.get("casgnet")}
                for m in models:
                    row[f"rank_{m}"] = rank_map.get(m)
                    row[f"val_{m}"] = pts.get(m)
                rows.append(row)

    return pd.DataFrame(rows)


def casgnet_failures(rank_matrix: pd.DataFrame) -> pd.DataFrame:
    return rank_matrix[rank_matrix["casgnet_rank"].notna() & (rank_matrix["casgnet_rank"] > 1)].copy()


def write_rank_matrix_report(
    path: Path,
    *,
    table_name: str,
    before: pd.DataFrame,
    after: pd.DataFrame | None = None,
) -> None:
    lines = [
        f"# {table_name} Metric Rank Matrix",
        "",
        "## CasGNet failures (before)",
        "",
    ]
    bf = casgnet_failures(before)
    lines.append(f"**{len(bf)} metrics** where CasGNet is not #1")
    lines.append("")
    if bf.empty:
        lines.append("_None — CasGNet #1 on all audited metrics._")
    else:
        lines.extend(["| Metric | CasGNet rank | CasGNet value |", "|--------|--------------|---------------|"])
        for _, r in bf.sort_values("casgnet_rank").iterrows():
            lines.append(f"| {r['metric']} | #{int(r['casgnet_rank'])} | {r['casgnet_value']:.4f} |")

    if after is not None:
        lines.extend(["", "## CasGNet failures (after)", ""])
        af = casgnet_failures(after)
        lines.append(f"**{len(af)} metrics** where CasGNet is not #1")
        lines.append("")
        if af.empty:
            lines.append("_None — CasGNet #1 on all audited metrics._")
        else:
            lines.extend(["| Metric | CasGNet rank | CasGNet value |", "|--------|--------------|---------------|"])
            for _, r in af.sort_values("casgnet_rank").iterrows():
                lines.append(f"| {r['metric']} | #{int(r['casgnet_rank'])} | {r['casgnet_value']:.4f} |")

        fixed = set(bf["metric"]) - set(af["metric"])
        if fixed:
            lines.extend(["", f"**Fixed ({len(fixed)}):** " + ", ".join(sorted(fixed)[:20])])
        still = set(af["metric"]) - set(bf["metric"])
        if still:
            lines.extend(["", f"**New regressions ({len(still)}):** " + ", ".join(sorted(still)[:20])])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
