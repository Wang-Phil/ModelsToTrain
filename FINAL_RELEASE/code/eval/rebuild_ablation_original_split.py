#!/usr/bin/env python3
"""Re-evaluate the 8 CASGNet SA/GRN/SK-UNIT ablation variants on the ORIGINAL
test split (old_data/test, n=258) for consistency with the original-split
Table 1.

This script does NOT modify the existing subset217 (n=230) ablation files
(ABLATION_SUMMARY.csv, ABLATION_RESULTS.xlsx, per_model/*/metrics.json,
test_roc.png, test_confusion.png). Instead it writes *_original.json /
*_original.csv / *_original.png siblings and a new
ABLATION_SUMMARY_ORIGINAL.csv + ABLATION_RESULTS_ORIGINAL.xlsx package.

Source data:
- casgnet_full (ab111)  -> reused from table1_final_package_original/casgnet
- starnet_s1_baseline (ab000) -> reused from table1_final_package_original/starnet_s1
- 6 middle variants -> per_model/{variant}/pool_predictions.npz filtered to
  split_tags == 'test' (old_data/test, n=258). If the npz is missing, the
  variant is skipped and reported.

Usage (project root):
  python evaluation_results/excel_aligned/rebuild_ablation_original_split.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from train_casgnet_contrastive_newdata import (  # noqa: E402
    bootstrap_auc_ci,
    bootstrap_classification_metrics_ci,
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ABLATION_ROOT = HERE / "ablation"
PER_MODEL_DIR = ABLATION_ROOT / "per_model"
SUMMARY_CSV_ORIG = ABLATION_ROOT / "ABLATION_SUMMARY_ORIGINAL.csv"
RESULTS_XLSX_ORIG = ABLATION_ROOT / "ABLATION_RESULTS_ORIGINAL.xlsx"
T1_ORIG_ROOT = HERE / "table1_final_package_original"
T1_ORIG_PER_MODEL = T1_ORIG_ROOT / "per_model"
T1_SUMMARY_CSV = T1_ORIG_ROOT / "TABLE1_SUMMARY.csv"
T1_PER_CLASS_CSV = T1_ORIG_ROOT / "TABLE1_PER_CLASS.csv"

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# 8 ablation variants — must match build_ablation_package.py ordering.
VARIANTS: list[dict] = [
    {"variant": "casgnet_full", "ab_code": "ab111",
     "sa": True, "grn": True, "sk": True,
     "t1_model": "casgnet", "target_auc": 0.962},
    {"variant": "casgnet_no_sa", "ab_code": "ab011",
     "sa": False, "grn": True, "sk": True, "target_auc": 0.960},
    {"variant": "casgnet_no_grn", "ab_code": "ab101",
     "sa": True, "grn": False, "sk": True, "target_auc": 0.955},
    {"variant": "casgnet_no_skunit", "ab_code": "ab110",
     "sa": True, "grn": True, "sk": False, "target_auc": 0.954},
    {"variant": "casgnet_only_sa", "ab_code": "ab100",
     "sa": True, "grn": False, "sk": False, "target_auc": 0.957},
    {"variant": "casgnet_only_skunit", "ab_code": "ab001",
     "sa": False, "grn": False, "sk": True, "target_auc": 0.952},
    {"variant": "casgnet_only_grn", "ab_code": "ab010",
     "sa": False, "grn": True, "sk": False, "target_auc": 0.950},
    {"variant": "starnet_s1_baseline", "ab_code": "ab000",
     "sa": False, "grn": False, "sk": False,
     "t1_model": "starnet_s1", "target_auc": 0.943},
]

CLASS_ORDER = [
    "Acetabular Loosening", "Dislocation", "Fracture", "Good Place",
    "Spacer", "Stem Loosening", "Wear",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fmt_ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.3f}({lo:.3f}-{hi:.3f})"


def parse_ci(s: str) -> tuple[float, float, float]:
    """Parse 'mean(lo-hi)' string from TABLE1 CSVs."""
    mean_part, rest = s.split("(", 1)
    lo_part, hi_part = rest.rstrip(")").split("-", 1)
    return float(mean_part), float(lo_part), float(hi_part)


def load_t1_summary() -> dict[str, dict]:
    df = pd.read_csv(T1_SUMMARY_CSV)
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        out[str(r["model"])] = r.to_dict()
    return out


def load_t1_per_class() -> dict[str, list[dict]]:
    df = pd.read_csv(T1_PER_CLASS_CSV)
    out: dict[str, list[dict]] = {}
    for model, g in df.groupby("model"):
        out[str(model)] = g.to_dict("records")
    return out


def compute_metrics_with_ci(
    yt: np.ndarray, yhat: np.ndarray, probs: np.ndarray, n_cls: int,
) -> dict:
    point_auc = float(compute_macro_auc_ovr(yt, probs))
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(
        yt, probs, n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED
    )
    cls_boot = bootstrap_classification_metrics_ci(
        yt, yhat, n_classes=n_cls, n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED,
    )
    macro_pt, per_class = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)

    def _cell(key: str) -> tuple[float, float, float]:
        d = cls_boot.get(key, {}) or {}
        return (
            float(d.get("mean", macro_pt[key])),
            float(d.get("ci95_low", macro_pt[key])),
            float(d.get("ci95_high", macro_pt[key])),
        )

    s_m, s_lo, s_hi = _cell("sensitivity")
    sp_m, sp_lo, sp_hi = _cell("specificity")
    npv_m, npv_lo, npv_hi = _cell("npv")
    ppv_m, ppv_lo, ppv_hi = _cell("ppv")
    acc_m, acc_lo, acc_hi = _cell("acc")

    return {
        "auc_point": point_auc,
        "auc_ci": fmt_ci(auc_mean, auc_lo, auc_hi),
        "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "sensitivity": fmt_ci(s_m, s_lo, s_hi),
        "specificity": fmt_ci(sp_m, sp_lo, sp_hi),
        "npv": fmt_ci(npv_m, npv_lo, npv_hi),
        "ppv": fmt_ci(ppv_m, ppv_lo, ppv_hi),
        "acc": fmt_ci(acc_m, acc_lo, acc_hi),
        "point": {
            "auc": point_auc,
            "sensitivity": float(macro_pt["sensitivity"]),
            "specificity": float(macro_pt["specificity"]),
            "npv": float(macro_pt["npv"]),
            "ppv": float(macro_pt["ppv"]),
            "acc": float(macro_pt["acc"]),
        },
        "per_class": per_class,
        "n_bootstrap": N_BOOTSTRAP,
        "n_samples": int(len(yt)),
    }


def per_class_auc_rows(
    yt: np.ndarray, yhat: np.ndarray, probs: np.ndarray,
    class_names: list[str], variant: str, ab_code: str,
) -> list[dict]:
    """Per-class metrics with CI for a middle variant on the test split."""
    n_cls = len(class_names)
    rows: list[dict] = []
    for c in range(n_cls):
        y_bin = (yt == c).astype(np.int32)
        n_pos = int(y_bin.sum())
        n_neg = len(yt) - n_pos
        name = class_names[c]
        # Per-class AUC with bootstrap CI
        if n_pos > 0 and n_neg > 0:
            pc_auc_pt = float(roc_auc_score(y_bin, probs[:, c]))
            rng = np.random.default_rng(BOOTSTRAP_SEED)
            boots = []
            idx_pos = np.where(y_bin == 1)[0]
            idx_neg = np.where(y_bin == 0)[0]
            for _ in range(N_BOOTSTRAP):
                sp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
                sn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
                samp = np.concatenate([sp, sn])
                yb = y_bin[samp]
                pb = probs[samp, c]
                if len(np.unique(yb)) < 2:
                    continue
                boots.append(float(roc_auc_score(yb, pb)))
            if boots:
                auc_lo = float(np.percentile(boots, 2.5))
                auc_hi = float(np.percentile(boots, 97.5))
                auc_ci = fmt_ci(pc_auc_pt, auc_lo, auc_hi)
            else:
                auc_ci = fmt_ci(pc_auc_pt, pc_auc_pt, pc_auc_pt)
        else:
            pc_auc_pt = float("nan")
            auc_ci = ""

        # Confusion-matrix-based per-class metrics with bootstrap CI
        cm = confusion_matrix(yt, yhat, labels=np.arange(n_cls))
        N = cm.sum()
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = N - tp - fn - fp

        def safe(num, den):
            return float(num) / float(den) if den > 0 else float("nan")

        pt_metrics = {
            "sensitivity": safe(tp, tp + fn),
            "specificity": safe(tn, tn + fp),
            "ppv": safe(tp, tp + fp),
            "npv": safe(tn, tn + fn),
            "acc": safe(tp + tn, tp + tn + fp + fn),
        }

        # Bootstrap per-class metrics
        rng = np.random.default_rng(BOOTSTRAP_SEED + c)
        boot_mets = {k: [] for k in pt_metrics}
        for _ in range(N_BOOTSTRAP):
            samp = rng.choice(len(yt), size=len(yt), replace=True)
            ys, ys_hat = yt[samp], yhat[samp]
            cm_b = confusion_matrix(ys, ys_hat, labels=np.arange(n_cls))
            Nb = cm_b.sum()
            tp_b = cm_b[c, c]
            fn_b = cm_b[c, :].sum() - tp_b
            fp_b = cm_b[:, c].sum() - tp_b
            tn_b = Nb - tp_b - fn_b - fp_b
            for k, (num, den) in {
                "sensitivity": (tp_b, tp_b + fn_b),
                "specificity": (tn_b, tn_b + fp_b),
                "ppv": (tp_b, tp_b + fp_b),
                "npv": (tn_b, tn_b + fn_b),
                "acc": (tp_b + tn_b, tp_b + tn_b + fp_b + fn_b),
            }.items():
                if den > 0:
                    boot_mets[k].append(num / den)

        ci_cells = {}
        for k, pt in pt_metrics.items():
            if boot_mets[k]:
                lo = float(np.percentile(boot_mets[k], 2.5))
                hi = float(np.percentile(boot_mets[k], 97.5))
                ci_cells[k] = fmt_ci(pt, lo, hi)
            else:
                ci_cells[k] = fmt_ci(pt, pt, pt) if np.isfinite(pt) else ""

        rows.append({
            "variant": variant,
            "ab_code": ab_code,
            "class": name,
            "n": n_pos,
            "auc": auc_ci,
            "sensitivity": ci_cells["sensitivity"],
            "specificity": ci_cells["specificity"],
            "ppv": ci_cells["ppv"],
            "npv": ci_cells["npv"],
            "acc": ci_cells["acc"],
        })
    return rows


def plot_roc(probs: np.ndarray, yt: np.ndarray, class_names: list[str],
             out_path: Path, title_suffix: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 8))
    rows = []
    for c, name in enumerate(class_names):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        auc = float(roc_auc_score(y_bin, probs[:, c]))
        rows.append((c, name, auc))
    for i, (c, name, auc) in enumerate(rows):
        y_bin = (yt == c).astype(np.int32)
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        ax.plot(fpr, tpr, lw=1.8, color=cmap(i % 10),
                label=f"{name} (AUC={auc:.3f})")
    macro_auc = float(compute_macro_auc_ovr(yt, probs))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    title = f"ROC (macro OvR AUC = {macro_auc:.3f})"
    if title_suffix:
        title += f"  {title_suffix}"
    ax.set(xlim=(0, 1), ylim=(0, 1.05),
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=title)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(yt: np.ndarray, yhat: np.ndarray, class_names: list[str],
                   out_path: Path, title_suffix: str = "") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    cm = confusion_matrix(yt, yhat, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix (row-normalized)" + title_suffix)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=9, color=tc)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-variant processing
# --------------------------------------------------------------------------- #
def process_endpoint_variant(v: dict, t1_summary: dict, t1_per_class: dict,
                             summary_rows: list[dict],
                             per_class_rows: list[dict]) -> None:
    """casgnet_full / starnet_s1_baseline: copy from Table1 original."""
    name = v["variant"]
    t1_model = v["t1_model"]
    vdir = PER_MODEL_DIR / name
    vdir.mkdir(parents=True, exist_ok=True)

    summ = t1_summary[t1_model]
    # Parse CI cells
    auc_m, auc_lo, auc_hi = parse_ci(summ["auc"])
    s_m, s_lo, s_hi = parse_ci(summ["sensitivity"])
    sp_m, sp_lo, sp_hi = parse_ci(summ["specificity"])
    npv_m, npv_lo, npv_hi = parse_ci(summ["npv"])
    ppv_m, ppv_lo, ppv_hi = parse_ci(summ["ppv"])
    acc_m, acc_lo, acc_hi = parse_ci(summ["acc"])

    metrics = {
        "auc_point": float(summ["auc_point"]),
        "auc_ci": summ["auc"],
        "auc_mean": auc_m, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "sensitivity": summ["sensitivity"],
        "specificity": summ["specificity"],
        "npv": summ["npv"],
        "ppv": summ["ppv"],
        "acc": summ["acc"],
        "point": {
            "auc": float(summ["auc_point"]),
            "sensitivity": float(summ["sensitivity_point"]),
            "specificity": float(summ["specificity_point"]),
            "npv": float(summ["npv_point"]),
            "ppv": float(summ["ppv_point"]),
            "acc": float(summ["acc_point"]),
        },
        "source": f"table1_final_package_original/{t1_model}",
        "n_samples": int(summ["n_samples"]),
        "class_counts": summ["class_counts"],
        "n_bootstrap": N_BOOTSTRAP,
    }
    (vdir / "metrics_original.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Per-class CSV from TABLE1_PER_CLASS.csv
    pc_rows_t1 = t1_per_class[t1_model]
    pc_out = []
    for pc in pc_rows_t1:
        pc_out.append({
            "variant": name,
            "ab_code": v["ab_code"],
            "class": pc["class"],
            "n": _class_count_from_string(summ["class_counts"], pc["class"]),
            "auc": pc["auc"],
            "sensitivity": pc["sensitivity"],
            "specificity": pc["specificity"],
            "ppv": pc["ppv"],
            "npv": pc["npv"],
            "acc": pc["acc"],
        })
    pd.DataFrame(pc_out).to_csv(vdir / "metrics_per_class_original.csv", index=False)

    # Copy plots from Table1 per_model dir
    src_roc = T1_ORIG_PER_MODEL / t1_model / "test_roc.png"
    src_cm = T1_ORIG_PER_MODEL / t1_model / "test_confusion.png"
    dst_roc = vdir / "test_roc_original.png"
    dst_cm = vdir / "test_confusion_original.png"
    if src_roc.is_file():
        dst_roc.write_bytes(src_roc.read_bytes())
    if src_cm.is_file():
        dst_cm.write_bytes(src_cm.read_bytes())

    summary_rows.append({
        "variant": name,
        "ab_code": v["ab_code"],
        "SA": "√" if v["sa"] else "×",
        "GRN": "√" if v["grn"] else "×",
        "SK_UNIT": "√" if v["sk"] else "×",
        "AUC": metrics["auc_ci"],
        "SENSITIVITY": metrics["sensitivity"],
        "SPECIFICITY": metrics["specificity"],
        "NPV": metrics["npv"],
        "PPV": metrics["ppv"],
        "ACC": metrics["acc"],
        "n": int(summ["n_samples"]),
        "auc_point": round(metrics["point"]["auc"], 4),
        "target_auc": v["target_auc"],
        "auc_delta": round(metrics["point"]["auc"] - v["target_auc"], 4),
    })
    per_class_rows.extend(pc_out)
    print(f"  [{name}] endpoint (reused T1 original): AUC={metrics['auc_ci']}  "
          f"ACC={metrics['acc']}")


def _class_count_from_string(class_counts_str: str, cls_name: str) -> int:
    """Map a class name to its count from the JSON-style '{\"0\": 58, ...}' string.

    Uses CLASS_ORDER to find the index for cls_name.
    """
    try:
        d = json.loads(class_counts_str)
    except Exception:
        return -1
    if cls_name in CLASS_ORDER:
        idx = CLASS_ORDER.index(cls_name)
        return int(d.get(str(idx), -1))
    return -1


def process_middle_variant(v: dict, summary_rows: list[dict],
                           per_class_rows: list[dict]) -> bool:
    """6 middle variants: filter pool_predictions.npz to test split (n=258)."""
    name = v["variant"]
    vdir = PER_MODEL_DIR / name
    npz_path = vdir / "pool_predictions.npz"
    if not npz_path.is_file():
        print(f"  [{name}] !! pool_predictions.npz MISSING -> {npz_path}")
        return False

    d = np.load(npz_path, allow_pickle=True)
    probs = d["probs"]
    yt = d["yt"]
    yhat = d["yhat"]
    class_names = [str(x) for x in d["class_names"].tolist()]
    paths = [str(x) for x in d["paths"].tolist()]
    split_tags = [str(x) for x in d["split_tags"].tolist()] if "split_tags" in d else None

    # Filter to test split. Prefer split_tags == 'test'; fall back to path filter.
    if split_tags is not None:
        mask = np.array([t == "test" for t in split_tags])
    else:
        mask = np.array(["old_data/test" in p for p in paths])
    if mask.sum() == 0:
        print(f"  [{name}] !! no test split samples in pool (n_pool={len(yt)})")
        return False
    if mask.sum() != 258:
        print(f"  [{name}] WARN: test split n={int(mask.sum())} (expected 258)")

    yt_t = yt[mask]
    yhat_t = yhat[mask]
    probs_t = probs[mask]
    n_cls = len(class_names)

    metrics = compute_metrics_with_ci(yt_t, yhat_t, probs_t, n_cls)
    (vdir / "metrics_original.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    pc_rows = per_class_auc_rows(yt_t, yhat_t, probs_t, class_names, name, v["ab_code"])
    pd.DataFrame(pc_rows).to_csv(vdir / "metrics_per_class_original.csv", index=False)

    plot_roc(probs_t, yt_t, class_names, vdir / "test_roc_original.png",
             title_suffix=f"({name}, test n={len(yt_t)})")
    plot_confusion(yt_t, yhat_t, class_names, vdir / "test_confusion_original.png",
                   title_suffix=f" ({name})")

    summary_rows.append({
        "variant": name,
        "ab_code": v["ab_code"],
        "SA": "√" if v["sa"] else "×",
        "GRN": "√" if v["grn"] else "×",
        "SK_UNIT": "√" if v["sk"] else "×",
        "AUC": metrics["auc_ci"],
        "SENSITIVITY": metrics["sensitivity"],
        "SPECIFICITY": metrics["specificity"],
        "NPV": metrics["npv"],
        "PPV": metrics["ppv"],
        "ACC": metrics["acc"],
        "n": int(len(yt_t)),
        "auc_point": round(metrics["point"]["auc"], 4),
        "target_auc": v["target_auc"],
        "auc_delta": round(metrics["point"]["auc"] - v["target_auc"], 4),
    })
    per_class_rows.extend(pc_rows)
    print(f"  [{name}] middle (test n={len(yt_t)}): AUC={metrics['auc_ci']}  "
          f"ACC={metrics['acc']}  SENS={metrics['sensitivity']}")
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    print(f"Re-evaluating ablation on ORIGINAL test split (n=258)")
    print(f"  ablation root: {ABLATION_ROOT}")
    print(f"  T1 original:   {T1_ORIG_ROOT}")

    t1_summary = load_t1_summary()
    t1_per_class = load_t1_per_class()

    summary_rows: list[dict] = []
    per_class_rows: list[dict] = []
    missing_npz: list[str] = []
    filter_failed: list[str] = []

    for v in VARIANTS:
        if "t1_model" in v:
            process_endpoint_variant(v, t1_summary, t1_per_class,
                                      summary_rows, per_class_rows)
        else:
            ok = process_middle_variant(v, summary_rows, per_class_rows)
            if not ok:
                npz = (PER_MODEL_DIR / v["variant"] / "pool_predictions.npz")
                if not npz.is_file():
                    missing_npz.append(v["variant"])
                else:
                    filter_failed.append(v["variant"])

    # Preserve VARIANTS ordering
    order = {v["variant"]: i for i, v in enumerate(VARIANTS)}
    summary_rows.sort(key=lambda r: order.get(r["variant"], 99))
    per_class_rows.sort(key=lambda r: (order.get(r["variant"], 99),
                                        CLASS_ORDER.index(r["class"])
                                        if r["class"] in CLASS_ORDER else 99))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV_ORIG, index=False)
    print(f"\nWrote summary CSV -> {SUMMARY_CSV_ORIG}")

    per_class_df = pd.DataFrame(per_class_rows)
    with pd.ExcelWriter(RESULTS_XLSX_ORIG, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Overall", index=False)
        per_class_df.to_excel(xl, sheet_name="PerClass", index=False)
    print(f"Wrote Excel       -> {RESULTS_XLSX_ORIG}")

    # ------------------------------------------------------------------ #
    # Report: ablation logic check + comparison with subset217
    # ------------------------------------------------------------------ #
    subset_csv = ABLATION_ROOT / "ABLATION_SUMMARY.csv"
    subset_df = pd.read_csv(subset_csv)
    sub_auc = {r["variant"]: r["AUC"] for _, r in subset_df.iterrows()}

    print("\n=== AUC ranking (original test) ===")
    for r in sorted(summary_rows, key=lambda x: -x["auc_point"]):
        print(f"  {r['variant']:24s} ab={r['ab_code']}  "
              f"AUC={r['AUC']}  Δ(subset217 {sub_auc.get(r['variant'],'?')})")

    full = next(r for r in summary_rows if r["variant"] == "casgnet_full")
    base = next(r for r in summary_rows if r["variant"] == "starnet_s1_baseline")
    print(f"\nfull AUC={full['auc_point']:.4f}  baseline AUC={base['auc_point']:.4f}  "
          f"full>baseline? {full['auc_point'] > base['auc_point']}")

    print("\n=== ΔAUC vs subset217 (point estimates) ===")
    sub_point = {r["variant"]: float(r["auc_point"]) for _, r in subset_df.iterrows()}
    for r in summary_rows:
        old = sub_point.get(r["variant"])
        if old is None:
            continue
        print(f"  {r['variant']:24s}  orig={r['auc_point']:.4f}  "
              f"subset217={old:.4f}  Δ={r['auc_point']-old:+.4f}")

    if missing_npz:
        print(f"\nMISSING pool_predictions.npz: {missing_npz}")
    if filter_failed:
        print(f"\nFILTER FAILED (no test split): {filter_failed}")


if __name__ == "__main__":
    main()
