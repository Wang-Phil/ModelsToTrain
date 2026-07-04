"""Per-class AUC comparison bar chart: CasGNet vs StarNet (Table1, subset217).

Reads ``table1_final_package/TABLE1_PER_CLASS.csv`` (the canonical per-class
table reproduced from checkpoints).  For each of the 7 classes a grouped bar
pair shows CasGNet AUC vs StarNet AUC, with the gap (CasGNet - StarNet)
annotated on top of each pair.  Classes where CasGNet dominates by >= 0.05
are highlighted (orange edge + asterisk) and classes where StarNet wins are
annotated in red.

Standalone: every path is resolved relative to this file, so the script can be
run from anywhere as ``python plot_per_class_auc_bar.py``.

Outputs (in this script's own directory):
  - per_class_auc_bar.png            (vertical grouped bars, 300 dpi)
  - per_class_auc_bar_horizontal.png (horizontal grouped bars, 300 dpi)
  - per_class_auc_bar.pdf            (2-page PDF: vertical + horizontal)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ---------------------------------------------------------------------------
# Configuration (paths resolved from this file so it is portable).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# training_curves/casgnet_vs_starnet/ -> excel_aligned/ -> evaluation_results/
TABLE1_DIR = SCRIPT_DIR.parents[1] / "table1_final_package"
OUT_DIR = SCRIPT_DIR

PER_CLASS_CSV = TABLE1_DIR / "TABLE1_PER_CLASS.csv"
# Fallback: per-model test_roc_auc.csv files if the combined CSV is missing.
PER_MODEL_DIR = TABLE1_DIR / "per_model"

# Model name in TABLE1_PER_CLASS.csv -> our key.
MODEL_CSV_NAME = {"casgnet": "casgnet", "starnet": "starnet_s1"}
DISPLAY_NAME = {"casgnet": "CasGNet", "starnet": "StarNet"}
COLORS = {"casgnet": "#1f77b4", "starnet": "#d62728"}

# Display order (matches the user's spec).
CLASS_ORDER = [
    "Acetabular Loosening",
    "Dislocation",
    "Fracture",
    "Good Place",
    "Spacer",
    "Stem Loosening",
    "Wear",
]

# Highlight threshold: CasGNet wins by >= this much.
DOMINANCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_auc(cell: str) -> float:
    """Parse '0.959(0.902-1.000)' -> 0.959."""
    return float(cell.split("(")[0].strip())


def load_per_class_auc() -> Dict[str, Dict[str, float]]:
    """Return {model_key: {class_name: auc}}.

    Primary source: TABLE1_PER_CLASS.csv.
    Fallback: per_model/{model}/test_roc_auc.csv.
    """
    out: Dict[str, Dict[str, float]] = {"casgnet": {}, "starnet": {}}

    if PER_CLASS_CSV.exists():
        with open(PER_CLASS_CSV) as f:
            for row in csv.DictReader(f):
                m_csv = row["excel_model"]
                for key, csv_name in MODEL_CSV_NAME.items():
                    if m_csv == csv_name and row["split"] == "test":
                        out[key][row["class"]] = _parse_auc(row["auc"])
        if all(out[m] for m in out):
            return out

    # Fallback: per-model files.
    print("[WARN] TABLE1_PER_CLASS.csv incomplete; falling back to per_model files.")
    for key, csv_name in MODEL_CSV_NAME.items():
        path = PER_MODEL_DIR / csv_name / "test_roc_auc.csv"
        with open(path) as f:
            for row in csv.DictReader(f):
                out[key][row["CLASS"]] = _parse_auc(row["AUC"])
    return out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _short_class(name: str, width: int = 12) -> str:
    """Wrap class names onto two lines for the vertical bar chart."""
    if len(name) <= width:
        return name
    parts = name.split()
    if len(parts) == 2:
        return f"{parts[0]}\n{parts[1]}"
    return name


def _annotate_gap(ax, x: float, c_val: float, s_val: float, vertical: bool) -> None:
    """Draw a bracket and gap label between two bar tops."""
    gap = c_val - s_val
    if vertical:
        y_top = max(c_val, s_val) + 0.025
        ax.plot(
            [x - 0.2, x - 0.2, x + 0.2, x + 0.2],
            [y_top - 0.008, y_top, y_top, y_top - 0.008],
            color="black", linewidth=0.9, clip_on=False,
        )
        color = "#2c3e50" if gap >= 0 else COLORS["starnet"]
        sign = "+" if gap >= 0 else ""
        ax.text(
            x, y_top + 0.004, f"{sign}{gap:.3f}",
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=color, clip_on=False,
        )
    else:
        # Horizontal: bars grow along x; bracket sits to the right.
        x_top = max(c_val, s_val) + 0.025
        ax.plot(
            [x_top - 0.008, x_top, x_top, x_top - 0.008],
            [x - 0.2, x - 0.2, x + 0.2, x + 0.2],
            color="black", linewidth=0.9, clip_on=False,
        )
        color = "#2c3e50" if gap >= 0 else COLORS["starnet"]
        sign = "+" if gap >= 0 else ""
        ax.text(
            x_top + 0.004, x, f"{sign}{gap:.3f}",
            ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=color, clip_on=False,
        )


def _mark_dominance(ax, x: float, gap: float, vertical: bool) -> None:
    """Add an asterisk above heavily-dominating CasGNet bars."""
    if gap < DOMINANCE_THRESHOLD:
        return
    if vertical:
        ax.text(
            x, 1.005, "*", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=16, fontweight="bold",
            color="#ff7f0e",
        )
    else:
        ax.text(
            1.005, x, "*", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=16, fontweight="bold",
            color="#ff7f0e",
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_vertical(per_class: Dict[str, Dict[str, float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13, 6.5))

    n = len(CLASS_ORDER)
    x = np.arange(n)
    width = 0.38

    casg_vals = [per_class["casgnet"].get(c, float("nan")) for c in CLASS_ORDER]
    star_vals = [per_class["starnet"].get(c, float("nan")) for c in CLASS_ORDER]

    bars_c = ax.bar(
        x - width / 2, casg_vals, width,
        label=DISPLAY_NAME["casgnet"], color=COLORS["casgnet"],
        edgecolor="black", linewidth=0.7,
    )
    bars_s = ax.bar(
        x + width / 2, star_vals, width,
        label=DISPLAY_NAME["starnet"], color=COLORS["starnet"],
        edgecolor="black", linewidth=0.7,
    )

    # Value labels on top of each bar.
    for bars, vals in ((bars_c, casg_vals), (bars_s, star_vals)):
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.003,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8,
                color=bar.get_facecolor(),
            )

    # Gap annotations + dominance markers.
    for i, (c_val, s_val) in enumerate(zip(casg_vals, star_vals)):
        _annotate_gap(ax, i, c_val, s_val, vertical=True)
        _mark_dominance(ax, i, c_val - s_val, vertical=True)

    ax.set_xticks(x)
    ax.set_xticklabels([_short_class(c) for c in CLASS_ORDER], fontsize=9)
    ax.set_ylabel("Per-class AUC")
    ax.set_ylim(0.70, 1.07)
    ax.set_title(
        "Per-class AUC: CasGNet vs StarNet (Table1, subset217, n=230)\n"
        f"\"*\" marks classes where CasGNet wins by >= {DOMINANCE_THRESHOLD:.2f}",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_horizontal(per_class: Dict[str, Dict[str, float]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 7))

    n = len(CLASS_ORDER)
    y = np.arange(n)
    height = 0.38

    casg_vals = [per_class["casgnet"].get(c, float("nan")) for c in CLASS_ORDER]
    star_vals = [per_class["starnet"].get(c, float("nan")) for c in CLASS_ORDER]

    bars_c = ax.barh(
        y - height / 2, casg_vals, height,
        label=DISPLAY_NAME["casgnet"], color=COLORS["casgnet"],
        edgecolor="black", linewidth=0.7,
    )
    bars_s = ax.barh(
        y + height / 2, star_vals, height,
        label=DISPLAY_NAME["starnet"], color=COLORS["starnet"],
        edgecolor="black", linewidth=0.7,
    )

    for bars, vals in ((bars_c, casg_vals), (bars_s, star_vals)):
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                ha="left", va="center", fontsize=8,
                color=bar.get_facecolor(),
            )

    for i, (c_val, s_val) in enumerate(zip(casg_vals, star_vals)):
        _annotate_gap(ax, i, c_val, s_val, vertical=False)
        _mark_dominance(ax, i, c_val - s_val, vertical=False)

    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_ORDER, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Per-class AUC")
    ax.set_xlim(0.70, 1.10)
    ax.set_title(
        "Per-class AUC: CasGNet vs StarNet (Table1, subset217, n=230)\n"
        f"\"*\" marks classes where CasGNet wins by >= {DOMINANCE_THRESHOLD:.2f}",
        fontsize=12,
    )
    ax.grid(True, linestyle="--", alpha=0.4, axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_class = load_per_class_auc()

    print("=== Per-class AUC (Table1, subset217) ===")
    print(f"  {'class':22s}  {'CasGNet':>8s}  {'StarNet':>8s}  {'gap':>8s}")
    print("  " + "-" * 56)
    dominance = []
    for c in CLASS_ORDER:
        cv = per_class["casgnet"].get(c)
        sv = per_class["starnet"].get(c)
        if cv is None or sv is None:
            print(f"  {c:22s}  {'--':>8s}  {'--':>8s}  {'--':>8s}")
            continue
        gap = cv - sv
        marker = " *" if gap >= DOMINANCE_THRESHOLD else ("  " if gap >= 0 else " (StarNet wins)")
        print(f"  {c:22s}  {cv:8.3f}  {sv:8.3f}  {gap:+8.3f}{marker}")
        dominance.append((c, gap))
    print()
    top3 = sorted(dominance, key=lambda t: -t[1])[:3]
    print("Top 3 CasGNet advantages:")
    for c, gap in top3:
        print(f"  {c:22s}  +{gap:.3f}")

    fig_v = plot_vertical(per_class)
    png_v = OUT_DIR / "per_class_auc_bar.png"
    fig_v.savefig(png_v, dpi=300, bbox_inches="tight")
    print(f"\n[OK] saved {png_v}")

    fig_h = plot_horizontal(per_class)
    png_h = OUT_DIR / "per_class_auc_bar_horizontal.png"
    fig_h.savefig(png_h, dpi=300, bbox_inches="tight")
    print(f"[OK] saved {png_h}")

    pdf_path = OUT_DIR / "per_class_auc_bar.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig_v, bbox_inches="tight")
        pdf.savefig(fig_h, bbox_inches="tight")
    print(f"[OK] saved {pdf_path}")

    plt.close(fig_v)
    plt.close(fig_h)
    print("\nDone.")


if __name__ == "__main__":
    main()
