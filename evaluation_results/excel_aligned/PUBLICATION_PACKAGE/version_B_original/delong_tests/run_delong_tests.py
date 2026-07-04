#!/usr/bin/env python3
"""DeLong test: casgnet vs 7 baselines on Version B original split caches.

Table1 (test, n=258): table1_final_package_original/caches/*_test_predictions.npz
Table2 (val, n=207):   table2_final_package_original/caches/*_val_predictions.npz

Macro-level comparison uses paired bootstrap on macro OvR AUC (n=2000, seed=42)
because no closed-form multiclass macro DeLong exists. Per-class sheets use
true binary one-vs-rest DeLong tests (DeLong et al. 1988 / fastDeLong).

Usage (project root):
  python evaluation_results/excel_aligned/run_delong_tests.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
# Support running from excel_aligned/ or delong_tests/ copy
EXCEL_ALIGNED = HERE.parent if HERE.name == "delong_tests" else HERE

# --------------------------------------------------------------------------- #
# Paths & config
# --------------------------------------------------------------------------- #
TABLE1_CACHE_DIR = HERE / "table1_final_package_original" / "caches"
TABLE2_CACHE_DIR = HERE / "table2_final_package_original" / "caches"
TABLE1_SUMMARY = HERE / "table1_final_package_original" / "TABLE1_SUMMARY.csv"
TABLE2_SUMMARY = HERE / "table2_final_package_original" / "TABLE2_SUMMARY.csv"
OUT_DIR = HERE / "delong_tests"
PUB_OUT_DIR = (
    HERE / "PUBLICATION_PACKAGE" / "version_B_original" / "delong_tests"
)

REFERENCE = "casgnet"
BASELINES = [
    "starnet",
    "lsnet_b",
    "densenet121",
    "resnet18",
    "resnet50",
    "mobilenetv4_m",
    "googlenet",
]

# checkpoint/cache filename stem -> display name
CACHE_NAME = {
    "casgnet": "casgnet",
    "starnet": "starnet_s1",
    "lsnet_b": "lsnet_b",
    "densenet121": "densenet121",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "mobilenetv4_m": "mobilenetv4_m",
    "googlenet": "googlenet",
}

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42
AUC_TOL = 0.005


def compute_macro_auc_ovr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro one-vs-rest AUC; skip classes with a single label in y_true."""
    n_classes = y_score.shape[1]
    aucs: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        try:
            a = float(roc_auc_score(y_bin, y_score[:, c]))
        except (ValueError, TypeError):
            continue
        if np.isfinite(a):
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else 0.0


# --------------------------------------------------------------------------- #
# DeLong (binary OVR) — fastDeLong / yandexdataschool roc_comparison
# --------------------------------------------------------------------------- #
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j_end = i
        while j_end < n and z[j_end] == z[i]:
            j_end += 1
        t[i:j_end] = 0.5 * (i + j_end - 1) + 1
        i = j_end
    t2 = np.empty(n, dtype=np.float64)
    t2[j] = t
    return t2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    pos = predictions_sorted_transposed[:, :m]
    neg = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(
    ground_truth: np.ndarray,
    predictions_one: np.ndarray,
    predictions_two: np.ndarray,
) -> tuple[float, float, float]:
    """Two-sided DeLong p-value comparing binary ROC AUCs of two score vectors."""
    gt = np.asarray(ground_truth, dtype=np.int32)
    s1 = np.asarray(predictions_one, dtype=np.float64)
    s2 = np.asarray(predictions_two, dtype=np.float64)

    if gt.size == 0 or np.unique(gt).size < 2:
        return float("nan"), float("nan"), float("nan")

    order = np.argsort(-gt)
    label_1_count = int(gt.sum())
    preds = np.vstack([s1, s2])[:, order]
    aucs, sigma = _fast_delong(preds, label_1_count)

    # logit transform for z-test (Hanley-McNeil / DeLong)
    aucs_clipped = np.clip(aucs, 1e-8, 1 - 1e-8)
    l = np.log(aucs_clipped / (1.0 - aucs_clipped))
    var = float(sigma[0, 0] + sigma[1, 1] - 2.0 * sigma[0, 1])
    if var <= 0 or not np.isfinite(var):
        return float("nan"), float(aucs[0]), float(aucs[1])
    z = (l[0] - l[1]) / np.sqrt(var)
    p = float(2.0 * norm.sf(abs(z)))
    return p, float(aucs[0]), float(aucs[1])


# --------------------------------------------------------------------------- #
# Metrics helpers
# --------------------------------------------------------------------------- #
def load_cache(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "yt": np.asarray(data["yt"]),
        "probs": np.asarray(data["probs"], dtype=np.float64),
        "class_names": list(data["class_names"]),
    }


def cache_path(cache_dir: Path, model: str, split_tag: str) -> Path:
    stem = CACHE_NAME[model]
    return cache_dir / f"{stem}_{split_tag}_predictions.npz"


def per_class_auc(yt: np.ndarray, probs: np.ndarray) -> dict[int, float]:
    n_cls = probs.shape[1]
    out: dict[int, float] = {}
    for c in range(n_cls):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        out[c] = float(roc_auc_score(y_bin, probs[:, c]))
    return out


def bootstrap_macro_aucs(
    yt: np.ndarray,
    probs: np.ndarray,
    boot_idx: np.ndarray,
) -> np.ndarray:
    """Bootstrap distribution of macro OvR AUC for one model."""
    n_boot, n_cls = boot_idx.shape[0], probs.shape[1]
    macro = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = boot_idx[b]
        yt_b = yt[idx]
        pr_b = probs[idx]
        aucs: list[float] = []
        for c in range(n_cls):
            y_bin = (yt_b == c).astype(np.int32)
            if np.unique(y_bin).size < 2:
                continue
            aucs.append(float(roc_auc_score(y_bin, pr_b[:, c])))
        macro[b] = float(np.mean(aucs)) if aucs else 0.0
    return macro


def paired_bootstrap_macro_p(diffs: np.ndarray) -> float:
    """Two-sided p-value from bootstrap macro AUC differences."""
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return min(p, 1.0)


def fisher_combine_pvalues(p_values: list[float]) -> float:
    """Fisher's method to combine independent p-values."""
    valid = [p for p in p_values if np.isfinite(p) and 0 < p <= 1]
    if not valid:
        return float("nan")
    stat = -2.0 * sum(np.log(p) for p in valid)
    return float(chi2.sf(stat, 2 * len(valid)))


def run_split(
    split_name: str,
    split_tag: str,
    cache_dir: Path,
    summary_csv: Path,
    expected_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ref_path = cache_path(cache_dir, REFERENCE, split_tag)
    ref = load_cache(ref_path)
    yt = ref["yt"]
    class_names = ref["class_names"]
    n_cls = ref["probs"].shape[1]

    if len(yt) != expected_n:
        raise ValueError(
            f"{split_name}: reference cache n={len(yt)}, expected {expected_n}"
        )

    ref_probs = ref["probs"]
    ref_macro = compute_macro_auc_ovr(yt, ref_probs)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot_idx = rng.integers(0, len(yt), size=(BOOTSTRAP_N, len(yt)))
    ref_boot = bootstrap_macro_aucs(yt, ref_probs, boot_idx)

    summary_rows = []
    per_class_rows = []

    for baseline in BASELINES:
        base_path = cache_path(cache_dir, baseline, split_tag)
        base = load_cache(base_path)
        if len(base["yt"]) != expected_n:
            raise ValueError(
                f"{split_name}/{baseline}: n={len(base['yt'])}, expected {expected_n}"
            )
        if not np.array_equal(base["yt"], yt):
            raise ValueError(f"{split_name}/{baseline}: y_true mismatch vs casgnet")

        base_probs = base["probs"]
        base_macro = compute_macro_auc_ovr(yt, base_probs)
        delta = ref_macro - base_macro

        class_pvals = []
        for c in range(n_cls):
            y_bin = (yt == c).astype(np.int32)
            p_delong, auc_ref_c, auc_base_c = delong_roc_test(
                y_bin, ref_probs[:, c], base_probs[:, c]
            )
            class_pvals.append(p_delong)
            per_class_rows.append(
                {
                    "split": split_name,
                    "baseline": baseline,
                    "class_idx": c,
                    "class_name": class_names[c],
                    "casgnet_class_auc": auc_ref_c,
                    "baseline_class_auc": auc_base_c,
                    "delta_class_auc": (
                        auc_ref_c - auc_base_c
                        if np.isfinite(auc_ref_c) and np.isfinite(auc_base_c)
                        else np.nan
                    ),
                    "delong_p_value": p_delong,
                    "significant_0.05": bool(p_delong < 0.05)
                    if np.isfinite(p_delong)
                    else False,
                }
            )

        boot_diffs = ref_boot - bootstrap_macro_aucs(yt, base_probs, boot_idx)
        bootstrap_p = paired_bootstrap_macro_p(boot_diffs)
        fisher_p = fisher_combine_pvalues(class_pvals)

        summary_rows.append(
            {
                "split": split_name,
                "baseline": baseline,
                "casgnet_macro_auc": round(ref_macro, 6),
                "baseline_macro_auc": round(base_macro, 6),
                "delta_auc": round(delta, 6),
                "delong_p_value": bootstrap_p,
                "fisher_combined_delong_p": fisher_p,
                "significant_0.05": bootstrap_p < 0.05,
                "casgnet_wins_auc": delta > 0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)

    # verify against TABLE summary
    verify = {"split": split_name, "reference_macro_auc": ref_macro, "checks": []}
    if summary_csv.exists():
        ref_row = pd.read_csv(summary_csv)
        casg_row = ref_row[ref_row["model"] == REFERENCE]
        if len(casg_row):
            expected = float(casg_row.iloc[0]["auc_point"])
            diff = abs(ref_macro - expected)
            verify["expected_casgnet_auc"] = expected
            verify["auc_diff"] = diff
            verify["auc_ok"] = diff <= AUC_TOL
            verify["checks"].append(
                {
                    "model": REFERENCE,
                    "computed": ref_macro,
                    "expected": expected,
                    "diff": diff,
                    "ok": diff <= AUC_TOL,
                }
            )
        for baseline in BASELINES:
            b_row = ref_row[ref_row["model"].isin([baseline, CACHE_NAME.get(baseline, baseline)])]
            if len(b_row) == 0 and baseline == "starnet":
                b_row = ref_row[ref_row["model"] == "starnet_s1"]
            if len(b_row):
                expected_b = float(b_row.iloc[0]["auc_point"])
                computed_b = float(
                    summary_df.loc[summary_df["baseline"] == baseline, "baseline_macro_auc"].iloc[0]
                )
                diff_b = abs(computed_b - expected_b)
                verify["checks"].append(
                    {
                        "model": baseline,
                        "computed": computed_b,
                        "expected": expected_b,
                        "diff": diff_b,
                        "ok": diff_b <= AUC_TOL,
                    }
                )

    return summary_df, per_class_df, verify


def write_excel(
    path: Path,
    t1_summary: pd.DataFrame,
    t2_summary: pd.DataFrame,
    t1_per_class: pd.DataFrame,
    t2_per_class: pd.DataFrame,
) -> None:
    export_t1 = t1_summary.drop(columns=["split"], errors="ignore")
    export_t2 = t2_summary.drop(columns=["split"], errors="ignore")

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        export_t1.to_excel(writer, sheet_name="Table1_Summary", index=False)
        export_t2.to_excel(writer, sheet_name="Table2_Summary", index=False)
        t1_per_class.to_excel(writer, sheet_name="Table1_PerClass", index=False)
        t2_per_class.to_excel(writer, sheet_name="Table2_PerClass", index=False)


def copy_outputs_to_publication() -> None:
    if not PUB_OUT_DIR.parent.exists():
        return
    PUB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "DELONG_TABLE1_TEST.csv",
        "DELONG_TABLE2_VAL.csv",
        "DELONG_RESULTS.xlsx",
        "DELONG_PER_CLASS_TABLE1.csv",
        "DELONG_PER_CLASS_TABLE2.csv",
        "run_delong_tests.py",
        "verification.json",
    ]:
        src = OUT_DIR / name
        if src.exists():
            shutil.copy2(src, PUB_OUT_DIR / name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Cache dirs:", flush=True)
    print(f"  Table1 (test): {TABLE1_CACHE_DIR}", flush=True)
    print(f"  Table2 (val):  {TABLE2_CACHE_DIR}", flush=True)

    print("Running Table1 (test)...", flush=True)
    t1_summary, t1_per_class, t1_verify = run_split(
        "test", "test", TABLE1_CACHE_DIR, TABLE1_SUMMARY, expected_n=258
    )
    print("Running Table2 (val)...", flush=True)
    t2_summary, t2_per_class, t2_verify = run_split(
        "val", "val", TABLE2_CACHE_DIR, TABLE2_SUMMARY, expected_n=207
    )

    # CSV outputs (summary columns per spec; fisher is extra in xlsx)
    t1_export = t1_summary[
        [
            "baseline",
            "casgnet_macro_auc",
            "baseline_macro_auc",
            "delta_auc",
            "delong_p_value",
            "significant_0.05",
            "casgnet_wins_auc",
        ]
    ]
    t2_export = t2_summary[
        [
            "baseline",
            "casgnet_macro_auc",
            "baseline_macro_auc",
            "delta_auc",
            "delong_p_value",
            "significant_0.05",
            "casgnet_wins_auc",
        ]
    ]

    t1_export.to_csv(OUT_DIR / "DELONG_TABLE1_TEST.csv", index=False)
    t2_export.to_csv(OUT_DIR / "DELONG_TABLE2_VAL.csv", index=False)
    t1_per_class.to_csv(OUT_DIR / "DELONG_PER_CLASS_TABLE1.csv", index=False)
    t2_per_class.to_csv(OUT_DIR / "DELONG_PER_CLASS_TABLE2.csv", index=False)
    write_excel(
        OUT_DIR / "DELONG_RESULTS.xlsx",
        t1_summary,
        t2_summary,
        t1_per_class,
        t2_per_class,
    )

    verification = {
        "table1": t1_verify,
        "table2": t2_verify,
        "cache_dirs": {
            "table1": str(TABLE1_CACHE_DIR),
            "table2": str(TABLE2_CACHE_DIR),
        },
        "note": (
            "Summary delong_p_value = paired bootstrap p-value for macro OvR AUC "
            f"(n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}). Per-class delong_p_value "
            "uses true binary OVR DeLong test."
        ),
    }
    (OUT_DIR / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )

    # copy script for reproducibility
    shutil.copy2(__file__, OUT_DIR / "run_delong_tests.py")

    copy_outputs_to_publication()

    print("\n=== Table1 (test) summary ===")
    print(t1_export.to_string(index=False))
    print("\n=== Table2 (val) summary ===")
    print(t2_export.to_string(index=False))
    print("\n=== AUC verification ===")
    for key in ("table1", "table2"):
        v = verification[key]
        print(f"  {key}: casgnet macro AUC={v['reference_macro_auc']:.4f}, ok={v.get('auc_ok')}")
        for c in v["checks"]:
            status = "OK" if c["ok"] else "MISMATCH"
            print(
                f"    {c['model']}: computed={c['computed']:.4f} "
                f"expected={c['expected']:.4f} diff={c['diff']:.4f} [{status}]"
            )
    print(f"\nOutputs written to: {OUT_DIR}")
    if PUB_OUT_DIR.exists():
        print(f"Copied to: {PUB_OUT_DIR}")


if __name__ == "__main__":
    main()
