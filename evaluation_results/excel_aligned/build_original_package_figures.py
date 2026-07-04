#!/usr/bin/env python3
"""Generate publication figures for the *original-split* packages.

Targets:
  evaluation_results/excel_aligned/table1_final_package_original/figures/
  evaluation_results/excel_aligned/table2_final_package_original/figures/

Outputs (per package figures/):
  - confusion_matrices_grid.png  — 2x4 grid of normalized confusion matrices (per-model PNGs)
  - overall_model_comparison.png — grouped bar chart of AUC / Sensitivity / Specificity / NPV / PPV / Accuracy
  - auc_bar.png                  — macro AUC bars with 95% CI error bars
  - README.md                    — figure listing

Data sources (no inference re-run, no manifest edits):
  - {package}/TABLE{1,2}_SUMMARY.csv         (overall metrics with CI)
  - {package}/per_model/{model}/{cm_name}    (existing per-model confusion PNG)

Note: original-split packages use field names `auc`, `sensitivity`, `specificity`, `npv`, `ppv`, `acc`
(string with CI) and `auc_point`, `acc_point`, ... (numeric point estimates). This is different from
the searched package which uses `repro_auc`/`repro_acc`. This script reads the original schema directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXCEL_ALIGNED = HERE
ROOT = EXCEL_ALIGNED.parents[1]

MODEL_LABELS = {
    "casgnet": "CasGNet",
    "starnet_s1": "StarNet",
    "lsnet_b": "LSNet-B",
    "densenet121": "DenseNet121",
    "resnet18": "ResNet18",
    "resnet50": "ResNet50",
    "googlenet": "GoogLeNet",
    "mobilenetv4_m": "MobileNetV4",
}

PACKAGE_CONFIG = {
    "table1": {
        "dir": EXCEL_ALIGNED / "table1_final_package_original",
        "summary_csv": "TABLE1_SUMMARY.csv",
        "split_label": "Table 1 (Test, original split, n=258)",
        "cm_name": "test_confusion.png",
        "roc_name": "test_roc.png",
    },
    "table2": {
        "dir": EXCEL_ALIGNED / "table2_final_package_original",
        "summary_csv": "TABLE2_SUMMARY.csv",
        "split_label": "Table 2 (Val, original split, n=207)",
        "cm_name": "val_confusion.png",
        "roc_name": "val_roc.png",
    },
}

METRIC_FIELDS = ["auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
METRIC_LABELS = {
    "auc": "AUC",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "npv": "NPV",
    "ppv": "PPV",
    "acc": "Accuracy",
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def parse_metric(s) -> tuple[float, float, float]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan, np.nan, np.nan
    m = re.match(r"([\d.]+)(?:\(([\d.]+)-([\d.]+)\))?", str(s).strip())
    if not m:
        return np.nan, np.nan, np.nan
    pt = float(m.group(1))
    if m.group(2) and m.group(3):
        return pt, float(m.group(2)), float(m.group(3))
    return pt, pt, pt


def load_summary(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["dir"] / cfg["summary_csv"])
    df["label"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))
    # sort by auc_point (fall back to parsing if column missing)
    if "auc_point" not in df.columns:
        df["auc_point"] = df["auc"].map(lambda x: parse_metric(x)[0])
    df = df.sort_values("auc_point", ascending=False).reset_index(drop=True)
    return df


def _yerr(lo, hi, pt) -> np.ndarray:
    return np.vstack([np.asarray(pt) - np.asarray(lo),
                      np.asarray(hi) - np.asarray(pt)])


def plot_auc_bar(summary: pd.DataFrame, cfg: dict, out_dir: Path) -> Path:
    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    pts, los, his = [], [], []
    for _, r in summary.iterrows():
        p, lo, hi = parse_metric(r["auc"])
        pts.append(p); los.append(lo); his.append(hi)
    pts = np.array(pts); los = np.array(los); his = np.array(his)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.85, len(labels)))
    ax.bar(x, pts, 0.6, color=colors, edgecolor="white", linewidth=0.6,
           yerr=_yerr(los, his, pts), capsize=4,
           error_kw={"elinewidth": 1.0, "capthick": 1.0})
    for xi, pt, lo, hi in zip(x, pts, los, his):
        ax.text(xi, hi + 0.005, f"{pt:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Macro AUC (95% CI)")
    ax.set_title(f"Macro AUC with 95% CI — {cfg['split_label']}")
    ax.set_ylim(max(0.0, np.nanmin(los) - 0.05), min(1.0, np.nanmax(his) + 0.04))
    fig.tight_layout()
    out = out_dir / "auc_bar.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out


def plot_overall_comparison(summary: pd.DataFrame, cfg: dict, out_dir: Path) -> Path:
    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    n_metrics = len(METRIC_FIELDS)
    width = 0.13

    fig, ax = plt.subplots(figsize=(15, 6.5))
    cmap = plt.get_cmap("tab10")
    metric_colors = {f: cmap(i / max(1, n_metrics - 1)) for i, f in enumerate(METRIC_FIELDS)}

    for k, field in enumerate(METRIC_FIELDS):
        pts, los, his = [], [], []
        for _, r in summary.iterrows():
            p, lo, hi = parse_metric(r[field])
            pts.append(p); los.append(lo); his.append(hi)
        pts = np.array(pts, dtype=float); los = np.array(los, dtype=float); his = np.array(his, dtype=float)
        offset = (k - (n_metrics - 1) / 2) * width
        ax.bar(x + offset, pts, width, color=metric_colors[field],
               edgecolor="white", linewidth=0.4,
               label=METRIC_LABELS[field],
               yerr=_yerr(los, his, pts), capsize=2,
               error_kw={"elinewidth": 0.6, "capthick": 0.6, "alpha": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Metric value (point estimate, 95% CI)")
    ax.set_title(f"Overall Model Comparison — {cfg['split_label']}")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower center", ncol=n_metrics, framealpha=0.9, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    out = out_dir / "overall_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out


def plot_confusion_matrices_grid(summary: pd.DataFrame, cfg: dict, out_dir: Path) -> Path:
    n = len(summary)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (_, row) in enumerate(summary.iterrows()):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        model = row["model"]
        label = row["label"]
        img_path = cfg["dir"] / "per_model" / model / cfg["cm_name"]
        if img_path.is_file():
            ax.imshow(mpimg.imread(img_path))
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        auc_pt = parse_metric(row["auc"])[0]
        ax.set_title(f"{label}\nAUC={auc_pt:.3f}", fontsize=10)
        ax.axis("off")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"Confusion Matrices (normalized) — {cfg['split_label']}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = out_dir / "confusion_matrices_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")
    return out


def write_readme(cfg: dict, fig_paths: list[Path]) -> Path:
    out_dir = cfg["dir"] / "figures"
    readme = out_dir / "README.md"
    lines = [
        f"# Figures — {cfg['dir'].name}",
        "",
        f"Split: **{cfg['split_label']}**",
        "",
        "Generated by `build_original_package_figures.py` (2026-06-28).",
        "All metrics are read from `TABLE*_SUMMARY.csv`; no source data is modified.",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
    ]
    descs = {
        "confusion_matrices_grid.png": "2×4 grid of normalized per-model confusion matrices",
        "overall_model_comparison.png": "Grouped bar chart of AUC / Sensitivity / Specificity / NPV / PPV / Accuracy",
        "auc_bar.png": "Macro AUC bar chart with 95% CI error bars",
    }
    for p in fig_paths:
        rel = p.name
        desc = descs.get(rel, rel)
        lines.append(f"| `{rel}` | {desc} |")
    lines += ["", f"Total: {len(fig_paths)} figure(s)."]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] {readme}")
    return readme


def build_figures(table_key: str) -> list[Path]:
    cfg = PACKAGE_CONFIG[table_key]
    out_dir = cfg["dir"] / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(cfg)
    print(f"\n=== {table_key} ({cfg['dir'].name}) ===")
    print(f"  models: {len(summary)}  split: {cfg['split_label']}")

    paths = [
        plot_auc_bar(summary, cfg, out_dir),
        plot_overall_comparison(summary, cfg, out_dir),
        plot_confusion_matrices_grid(summary, cfg, out_dir),
    ]
    write_readme(cfg, paths)
    return paths


def main() -> None:
    apply_plot_style()
    all_paths: list[Path] = []
    for key in ("table1", "table2"):
        all_paths.extend(build_figures(key))
    print(f"\nGenerated {len(all_paths)} figure(s)")


if __name__ == "__main__":
    main()
