#!/usr/bin/env python3
"""
Generate package-level comparison figures for table1_final_package / table2_final_package.

Not invoked by default when building packages. Run standalone, or pass --figures to
build_table1_final_package.py / build_table2_final_package.py.

Outputs (per package figures/):
  - overall_model_comparison.png   — macro ACC + AUC bar charts
  - per_class_comparison.png       — per-class AUC heatmap (models × classes)
  - model_auc_comparison.png       — macro AUC bars + optional combined macro ROC
  - confusion_matrices_grid.png    — 2×4 grid of per-model confusion matrices

Usage (project root):
  python evaluation_results/excel_aligned/build_package_comparison_figures.py
  python evaluation_results/excel_aligned/build_package_comparison_figures.py --table table1
  python evaluation_results/excel_aligned/build_package_comparison_figures.py --table table2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    EXCEL_MODELS,
    bootstrap_per_class_auc_rows,
    load_cache,
    macro_metrics_row,
    parse_point,
)

MODELS = [m for m, _ in EXCEL_MODELS]

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

CLASS_ORDER = [
    "Acetabular Loosening",
    "Dislocation",
    "Fracture",
    "Good Place",
    "Spacer",
    "Stem Loosening",
    "Wear",
]

PACKAGE_CONFIG = {
    "table1": {
        "dir": HERE / "table1_final_package",
        "summary_csv": "TABLE1_SUMMARY.csv",
        "per_class_csv": "TABLE1_PER_CLASS.csv",
        "split": "test",
        "split_label": "Table 1 (Test)",
        "cache_dir": HERE / "table1_per_model" / "caches",
        "cache_suffix": "_test_predictions.npz",
        "roc_name": "test_roc.png",
        "cm_name": "test_confusion.png",
    },
    "table2": {
        "dir": HERE / "table2_final_package",
        "summary_csv": "TABLE2_SUMMARY.csv",
        "per_class_csv": "TABLE2_PER_CLASS.csv",
        "split": "val",
        "split_label": "Table 2 (Val)",
        "cache_dir": HERE / "caches",
        "cache_suffix": "_val_predictions.npz",
        "roc_name": "val_roc.png",
        "cm_name": "val_confusion.png",
    },
}


def parse_metric(s: str | float | None) -> tuple[float, float, float]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan, np.nan, np.nan
    text = str(s).strip()
    m = re.match(r"([\d.]+)(?:\(([\d.]+)-([\d.]+)\))?", text)
    if not m:
        return np.nan, np.nan, np.nan
    point = float(m.group(1))
    if m.group(2) and m.group(3):
        lo, hi = float(m.group(2)), float(m.group(3))
    else:
        lo, hi = point, point
    return point, lo, hi


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
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


def load_summary(package_dir: Path, summary_name: str) -> pd.DataFrame:
    df = pd.read_csv(package_dir / summary_name)
    df["label"] = df["model"].map(lambda m: MODEL_LABELS.get(m, m))
    df["auc_point"] = df["repro_auc"].map(lambda x: parse_metric(x)[0])
    df["acc_point"] = df["repro_acc"].map(lambda x: parse_metric(x)[0])
    df = df.sort_values("auc_point", ascending=False).reset_index(drop=True)
    return df


def load_per_class_frame(
    package_dir: Path,
    per_class_name: str,
    cfg: dict,
) -> pd.DataFrame:
    pc_path = package_dir / per_class_name
    df = pd.read_csv(pc_path) if pc_path.is_file() else pd.DataFrame()

    class_col = "class" if "class" in df.columns else "model"
    if not df.empty and class_col in df.columns and df[class_col].notna().sum() >= len(MODELS) * 3:
        out = df.copy()
        if class_col != "class":
            out = out.rename(columns={class_col: "class"})
        return out

    rows: list[dict] = []
    for model, ck_name in EXCEL_MODELS:
        mdir = package_dir / "per_model" / model
        mcsv = mdir / "metrics_per_class.csv"
        if mcsv.is_file():
            mdf = pd.read_csv(mcsv)
            if len(mdf) >= 5 and "class" in mdf.columns:
                for _, r in mdf.iterrows():
                    rows.append(
                        {
                            "excel_model": model,
                            "split": cfg["split"],
                            "class": r["class"],
                            "auc": r.get("auc", ""),
                            "sensitivity": r.get("sensitivity", ""),
                            "specificity": r.get("specificity", ""),
                        }
                    )
                continue

        cache = cfg["cache_dir"] / f"{model}{cfg['cache_suffix']}"
        if not cache.is_file():
            continue
        probs, yt, yhat, class_names = load_cache(cache)
        _, pc_rows = macro_metrics_row(
            model, ck_name, cfg["split"], yt, yhat, probs, len(class_names), class_names
        )
        for r in pc_rows:
            rows.append(
                {
                    "excel_model": model,
                    "split": cfg["split"],
                    "class": r.get("model", r.get("class", "")),
                    "auc": r.get("auc", ""),
                    "sensitivity": r.get("sensitivity", ""),
                    "specificity": r.get("specificity", ""),
                }
            )

    if not rows:
        raise FileNotFoundError(f"No per-class data for {package_dir}")
    return pd.DataFrame(rows)


def _yerr(lo: np.ndarray, hi: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.vstack([point - lo, hi - point])


def plot_overall_comparison(summary: pd.DataFrame, cfg: dict, out_dir: Path) -> Path:
    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    width = 0.55

    acc = summary["repro_acc"].map(lambda s: parse_metric(s))
    auc = summary["repro_auc"].map(lambda s: parse_metric(s))
    acc_pt = np.array([a[0] for a in acc])
    acc_lo = np.array([a[1] for a in acc])
    acc_hi = np.array([a[2] for a in acc])
    auc_pt = np.array([a[0] for a in auc])
    auc_lo = np.array([a[1] for a in auc])
    auc_hi = np.array([a[2] for a in auc])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    color_acc = "#4C72B0"
    color_auc = "#55A868"

    ax = axes[0]
    ax.bar(
        x,
        acc_pt,
        width,
        color=color_acc,
        edgecolor="white",
        linewidth=0.6,
        yerr=_yerr(acc_lo, acc_hi, acc_pt),
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Macro Accuracy")
    ax.set_title(f"(a) Overall Accuracy — {cfg['split_label']}")
    ax.set_ylim(max(0.0, min(acc_lo) - 0.05), min(1.0, max(acc_hi) + 0.02))

    ax = axes[1]
    ax.bar(
        x,
        auc_pt,
        width,
        color=color_auc,
        edgecolor="white",
        linewidth=0.6,
        yerr=_yerr(auc_lo, auc_hi, auc_pt),
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Macro AUC")
    ax.set_title(f"(b) Overall AUC — {cfg['split_label']}")
    ax.set_ylim(max(0.0, min(auc_lo) - 0.05), min(1.0, max(auc_hi) + 0.02))

    fig.suptitle(
        f"Overall Model Comparison (8 Models, sorted by AUC ↓)\n{cfg['split_label']}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "overall_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_per_class_comparison(pc_df: pd.DataFrame, summary: pd.DataFrame, cfg: dict, out_dir: Path) -> Path:
    model_order = summary["model"].tolist()
    labels = [MODEL_LABELS.get(m, m) for m in model_order]

    classes = [c for c in CLASS_ORDER if c in pc_df["class"].unique()]
    if not classes:
        classes = sorted(pc_df["class"].unique())

    matrix = np.full((len(classes), len(model_order)), np.nan)
    for j, model in enumerate(model_order):
        sub = pc_df.loc[pc_df["excel_model"] == model]
        for i, cls in enumerate(classes):
            row = sub.loc[sub["class"] == cls]
            if not row.empty:
                matrix[i, j] = parse_metric(row.iloc[0]["auc"])[0]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = LinearSegmentedColormap.from_list("auc", ["#f7fbff", "#6baed6", "#08306b"])
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Per-class AUC (point estimate)")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Model (sorted by macro AUC ↓)")
    ax.set_ylabel("Class")

    for i in range(len(classes)):
        for j in range(len(model_order)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > 0.82 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(f"Per-class AUC — {cfg['split_label']}")
    fig.tight_layout()
    out = out_dir / "per_class_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def macro_roc_curve(probs: np.ndarray, yt: np.ndarray, n_cls: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve

    yt_bin = np.zeros((len(yt), n_cls), dtype=np.int32)
    yt_bin[np.arange(len(yt)), yt] = 1
    fpr_list, tpr_list = [], []
    for c in range(n_cls):
        if np.unique(yt_bin[:, c]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(yt_bin[:, c], probs[:, c])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    if not fpr_list:
        return np.array([0, 1]), np.array([0, 1])
    all_fpr = np.unique(np.concatenate(fpr_list))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(fpr_list)
    return all_fpr, mean_tpr


def plot_model_auc_comparison(
    summary: pd.DataFrame,
    cfg: dict,
    out_dir: Path,
) -> Path:
    from sklearn.metrics import auc as sk_auc

    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    auc_pt = summary["auc_point"].to_numpy()
    auc_vals = summary["repro_auc"].map(lambda s: parse_metric(s))
    auc_lo = np.array([a[1] for a in auc_vals])
    auc_hi = np.array([a[2] for a in auc_vals])

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_roc = fig.add_subplot(gs[0, 1])

    colors = plt.get_cmap("tab10")(np.linspace(0, 0.85, len(labels)))
    ax_bar.bar(
        x,
        auc_pt,
        0.6,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        yerr=_yerr(auc_lo, auc_hi, auc_pt),
        capsize=3,
    )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=28, ha="right")
    ax_bar.set_ylabel("Macro AUC")
    ax_bar.set_title(f"(a) Macro AUC by Model")
    ax_bar.set_ylim(max(0.0, min(auc_lo) - 0.05), min(1.0, max(auc_hi) + 0.02))

    for idx, (model, label) in enumerate(zip(summary["model"], labels)):
        cache = cfg["cache_dir"] / f"{model}{cfg['cache_suffix']}"
        if not cache.is_file():
            continue
        probs, yt, yhat, class_names = load_cache(cache)
        fpr, tpr = macro_roc_curve(probs, yt, len(class_names))
        roc_auc = sk_auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2.0, color=colors[idx], label=f"{label} ({roc_auc:.3f})")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax_roc.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate")
    ax_roc.set_title("(b) Macro-averaged ROC (one-vs-rest)")
    ax_roc.legend(loc="lower right", fontsize=7, frameon=True)
    ax_roc.grid(True, alpha=0.3)

    fig.suptitle(
        f"Model AUC Comparison — {cfg['split_label']}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = out_dir / "model_auc_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrices_grid(summary: pd.DataFrame, cfg: dict, package_dir: Path, out_dir: Path) -> Path:
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
        img_path = package_dir / "per_model" / model / cfg["cm_name"]
        if img_path.is_file():
            ax.imshow(mpimg.imread(img_path))
        else:
            ax.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{label}\nAUC={parse_metric(row['repro_auc'])[0]:.3f}", fontsize=10)
        ax.axis("off")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Confusion Matrices (normalized) — {cfg['split_label']}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out = out_dir / "confusion_matrices_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def build_package_figures(table_key: str) -> list[Path]:
    cfg = PACKAGE_CONFIG[table_key]
    package_dir = cfg["dir"]
    out_dir = package_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(package_dir, cfg["summary_csv"])
    pc_df = load_per_class_frame(package_dir, cfg["per_class_csv"], cfg)

    paths = [
        plot_overall_comparison(summary, cfg, out_dir),
        plot_per_class_comparison(pc_df, summary, cfg, out_dir),
        plot_model_auc_comparison(summary, cfg, out_dir),
        plot_confusion_matrices_grid(summary, cfg, package_dir, out_dir),
    ]
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["table1", "table2", "both"], default="both")
    args = ap.parse_args()
    apply_plot_style()

    keys = ["table1", "table2"] if args.table == "both" else [args.table]
    all_paths: list[Path] = []
    for key in keys:
        print(f"\n=== {key} ===")
        paths = build_package_figures(key)
        all_paths.extend(paths)
        for p in paths:
            print(f"  {p}")

    print(f"\nGenerated {len(all_paths)} figure(s)")


if __name__ == "__main__":
    main()
