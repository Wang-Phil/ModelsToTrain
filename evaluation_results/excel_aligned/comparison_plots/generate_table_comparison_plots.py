#!/usr/bin/env python3
"""
Generate Table 1 vs Table 2 comparison plots for 8 models.

Data sources:
  - table1_final_package/TABLE1_SUMMARY.csv
  - table2_final_package/TABLE2_SUMMARY.csv
  - rank_snapshots/table1_after.csv, table2_after.csv

Usage (project root or this directory):
  python evaluation_results/excel_aligned/comparison_plots/generate_table_comparison_plots.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXCEL_ALIGNED = HERE.parent

TABLE1_CSV = EXCEL_ALIGNED / "table1_final_package" / "TABLE1_SUMMARY.csv"
TABLE2_CSV = EXCEL_ALIGNED / "table2_final_package" / "TABLE2_SUMMARY.csv"
RANK1_CSV = EXCEL_ALIGNED / "rank_snapshots" / "table1_after.csv"
RANK2_CSV = EXCEL_ALIGNED / "rank_snapshots" / "table2_after.csv"
OUT_DIR = HERE

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

COLOR_T1 = "#4C72B0"
COLOR_T2 = "#DD8452"
TABLE1_LABEL = "Table 1 (Test)"
TABLE2_LABEL = "Table 2 (Val)"


def parse_metric(s: str | float | None) -> tuple[float, float, float]:
    """Return (point, lower, upper) from strings like '0.907(0.889-0.924)'."""
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
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )


def load_comparison_frame() -> pd.DataFrame:
    t1 = pd.read_csv(TABLE1_CSV)
    t2 = pd.read_csv(TABLE2_CSV)

    rows: list[dict] = []
    for model in t1["model"].unique():
        r1 = t1.loc[t1["model"] == model].iloc[0]
        r2 = t2.loc[t2["model"] == model].iloc[0]

        acc1, acc1_lo, acc1_hi = parse_metric(r1["repro_acc"])
        acc2, acc2_lo, acc2_hi = parse_metric(r2["repro_acc"])
        auc1, auc1_lo, auc1_hi = parse_metric(r1["repro_auc"])
        auc2, auc2_lo, auc2_hi = parse_metric(r2["repro_auc"])

        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "t1_acc": acc1,
                "t1_acc_lo": acc1_lo,
                "t1_acc_hi": acc1_hi,
                "t2_acc": acc2,
                "t2_acc_lo": acc2_lo,
                "t2_acc_hi": acc2_hi,
                "t1_auc": auc1,
                "t1_auc_lo": auc1_lo,
                "t1_auc_hi": auc1_hi,
                "t2_auc": auc2,
                "t2_auc_lo": auc2_lo,
                "t2_auc_hi": auc2_hi,
                "t1_rank_acc": int(r1["rank"]) if pd.notna(r1["rank"]) else np.nan,
                "t2_rank_acc": int(r2["rank"]) if pd.notna(r2["rank"]) else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("t1_auc", ascending=False).reset_index(drop=True)

    r1 = pd.read_csv(RANK1_CSV)
    r2 = pd.read_csv(RANK2_CSV)
    auc_rank1 = _rank_row_to_dict(r1, "macro:auc", prefix="rank_")
    auc_rank2 = _rank_row_to_dict(r2, "macro:auc", prefix="rank_")
    acc_rank1 = _rank_row_to_dict(r1, "macro:acc", prefix="rank_")
    acc_rank2 = _rank_row_to_dict(r2, "macro:acc", prefix="rank_")

    df["t1_rank_auc"] = df["model"].map(auc_rank1)
    df["t2_rank_auc"] = df["model"].map(auc_rank2)
    df["t1_rank_acc_snap"] = df["model"].map(acc_rank1)
    df["t2_rank_acc_snap"] = df["model"].map(acc_rank2)
    return df


def _rank_row_to_dict(rank_df: pd.DataFrame, metric: str, prefix: str) -> dict[str, float]:
    row = rank_df.loc[rank_df["metric"] == metric]
    if row.empty:
        return {}
    row = row.iloc[0]
    out: dict[str, float] = {}
    for col in rank_df.columns:
        if col.startswith(prefix):
            model = col[len(prefix) :]
            val = row[col]
            if pd.notna(val) and val != "":
                out[model] = float(val)
    return out


def _yerr(lo: np.ndarray, hi: np.ndarray, point: np.ndarray) -> np.ndarray:
    lower = point - lo
    upper = hi - point
    return np.vstack([lower, upper])


def plot_grouped_bars(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    t1 = df[f"t1_{metric}"].to_numpy()
    t2 = df[f"t2_{metric}"].to_numpy()
    t1_lo = df[f"t1_{metric}_lo"].to_numpy()
    t1_hi = df[f"t1_{metric}_hi"].to_numpy()
    t2_lo = df[f"t2_{metric}_lo"].to_numpy()
    t2_hi = df[f"t2_{metric}_hi"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars1 = ax.bar(
        x - width / 2,
        t1,
        width,
        label=TABLE1_LABEL,
        color=COLOR_T1,
        edgecolor="white",
        linewidth=0.6,
        yerr=_yerr(t1_lo, t1_hi, t1),
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )
    bars2 = ax.bar(
        x + width / 2,
        t2,
        width,
        label=TABLE2_LABEL,
        color=COLOR_T2,
        edgecolor="white",
        linewidth=0.6,
        yerr=_yerr(t2_lo, t2_hi, t2),
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.set_axisbelow(True)

    if ylim:
        ax.set_ylim(*ylim)
    else:
        ymin = min(np.nanmin(t1_lo), np.nanmin(t2_lo)) - 0.03
        ymax = max(np.nanmax(t1_hi), np.nanmax(t2_hi)) + 0.02
        ax.set_ylim(max(0.0, ymin), min(1.0, ymax))

    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_rank_comparison(df: pd.DataFrame) -> Path:
    """Dot plot: rank in Table 1 vs Table 2 for ACC and AUC."""
    labels = df["label"].tolist()[::-1]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    panels = [
        ("acc", "Macro Accuracy Rank", "t1_rank_acc_snap", "t2_rank_acc_snap"),
        ("auc", "Macro AUC Rank", "t1_rank_auc", "t2_rank_auc"),
    ]

    for ax, (_, title, col_t1, col_t2) in zip(axes, panels):
        sub = df.iloc[::-1]
        r1 = sub[col_t1].to_numpy()
        r2 = sub[col_t2].to_numpy()

        ax.hlines(y, r1, r2, color="#BBBBBB", linewidth=2, zorder=1)
        ax.scatter(r1, y, s=90, color=COLOR_T1, label=TABLE1_LABEL, zorder=3, edgecolors="white", linewidths=0.8)
        ax.scatter(r2, y, s=90, color=COLOR_T2, label=TABLE2_LABEL, zorder=3, edgecolors="white", linewidths=0.8)

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Rank (1 = best)")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.set_xticks(np.arange(1, 9))
        ax.set_xlim(8.6, 0.4)
        ax.grid(axis="x")

        for yi, (a, b) in enumerate(zip(r1, r2)):
            delta = int(b - a)
            if delta == 0:
                txt = "0"
                color = "#666666"
            elif delta > 0:
                txt = f"↓{delta}"
                color = "#C44E52"
            else:
                txt = f"↑{-delta}"
                color = "#55A868"
            ax.text(8.35, yi, txt, va="center", ha="left", fontsize=9, color=color)

    handles = [
        mpatches.Patch(color=COLOR_T1, label=TABLE1_LABEL),
        mpatches.Patch(color=COLOR_T2, label=TABLE2_LABEL),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Ranking Comparison: Table 1 vs Table 2", y=1.06, fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "rank_comparison_t1_vs_t2.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_summary_figure(df: pd.DataFrame) -> Path:
    """2×2 summary: ACC bars, AUC bars, ACC rank, AUC rank."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    # (a) ACC grouped bars
    ax = axes[0, 0]
    t1 = df["t1_acc"].to_numpy()
    t2 = df["t2_acc"].to_numpy()
    ax.bar(x - width / 2, t1, width, label=TABLE1_LABEL, color=COLOR_T1, edgecolor="white")
    ax.bar(x + width / 2, t2, width, label=TABLE2_LABEL, color=COLOR_T2, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Macro Accuracy")
    ax.set_ylim(0.72, 0.96)
    ax.legend(loc="upper right", fontsize=9)

    # (b) AUC grouped bars
    ax = axes[0, 1]
    t1 = df["t1_auc"].to_numpy()
    t2 = df["t2_auc"].to_numpy()
    ax.bar(x - width / 2, t1, width, label=TABLE1_LABEL, color=COLOR_T1, edgecolor="white")
    ax.bar(x + width / 2, t2, width, label=TABLE2_LABEL, color=COLOR_T2, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Macro AUC")
    ax.set_title("(b) Macro AUC")
    ax.set_ylim(0.78, 0.98)
    ax.legend(loc="upper right", fontsize=9)

    # (c)(d) rank dot panels
    rank_panels = [
        (axes[1, 0], "t1_rank_acc_snap", "t2_rank_acc_snap", "(c) Accuracy Rank"),
        (axes[1, 1], "t1_rank_auc", "t2_rank_auc", "(d) AUC Rank"),
    ]
    y_labels = labels[::-1]
    y_pos = np.arange(len(y_labels))

    for ax, col_t1, col_t2, title in rank_panels:
        sub = df.iloc[::-1]
        r1 = sub[col_t1].to_numpy()
        r2 = sub[col_t2].to_numpy()
        ax.hlines(y_pos, r1, r2, color="#BBBBBB", linewidth=1.8, zorder=1)
        ax.scatter(r1, y_pos, s=70, color=COLOR_T1, zorder=3, edgecolors="white", linewidths=0.6)
        ax.scatter(r2, y_pos, s=70, color=COLOR_T2, zorder=3, edgecolors="white", linewidths=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Rank (1 = best)")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.set_xticks(np.arange(1, 9))
        ax.set_xlim(8.5, 0.5)

    fig.suptitle(
        "Table 1 vs Table 2 — Overall Model Comparison (8 Models)\n"
        "Sorted by Table 1 Macro AUC (descending)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "summary_t1_vs_t2_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_plotted_values_csv(df: pd.DataFrame) -> Path:
    out_cols = [
        "model",
        "label",
        "t1_acc",
        "t1_acc_lo",
        "t1_acc_hi",
        "t2_acc",
        "t2_acc_lo",
        "t2_acc_hi",
        "t1_auc",
        "t1_auc_lo",
        "t1_auc_hi",
        "t2_auc",
        "t2_auc_lo",
        "t2_auc_hi",
        "t1_rank_acc_snap",
        "t2_rank_acc_snap",
        "t1_rank_auc",
        "t2_rank_auc",
        "acc_delta_t2_minus_t1",
        "auc_delta_t2_minus_t1",
        "rank_acc_delta_t2_minus_t1",
        "rank_auc_delta_t2_minus_t1",
    ]
    export = df.copy()
    export["acc_delta_t2_minus_t1"] = export["t2_acc"] - export["t1_acc"]
    export["auc_delta_t2_minus_t1"] = export["t2_auc"] - export["t1_auc"]
    export["rank_acc_delta_t2_minus_t1"] = export["t2_rank_acc_snap"] - export["t1_rank_acc_snap"]
    export["rank_auc_delta_t2_minus_t1"] = export["t2_rank_auc"] - export["t1_rank_auc"]
    out = OUT_DIR / "plotted_values_t1_vs_t2.csv"
    export[out_cols].to_csv(out, index=False, float_format="%.4f")
    return out


def main() -> None:
    apply_plot_style()
    df = load_comparison_frame()

    paths = [
        plot_grouped_bars(
            df,
            metric="acc",
            ylabel="Macro Accuracy",
            title="Macro Accuracy: Table 1 (Test) vs Table 2 (Val)",
            filename="grouped_bar_acc_t1_vs_t2.png",
        ),
        plot_grouped_bars(
            df,
            metric="auc",
            ylabel="Macro AUC",
            title="Macro AUC: Table 1 (Test) vs Table 2 (Val)",
            filename="grouped_bar_auc_t1_vs_t2.png",
        ),
        plot_rank_comparison(df),
        plot_summary_figure(df),
        save_plotted_values_csv(df),
    ]

    print("Generated outputs:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
