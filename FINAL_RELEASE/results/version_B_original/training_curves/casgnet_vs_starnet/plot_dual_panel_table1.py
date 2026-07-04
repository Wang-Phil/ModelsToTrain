"""Dual-panel figure: validation AUC curves (left) + Table1 final AUC (right).

Left panel  : CasGNet vs StarNet val_auc over 200 epochs (from
              training_curves/data/{casgnet,starnet}_history.csv), with best
              val_auc points annotated (CasGNet 0.9449@ep157, StarNet 0.9351@ep198).
Right panel : Table1 (subset217, n=230) final macro-AUC bar chart, CasGNet 0.965
              vs StarNet 0.946, with the ~0.019 gap annotated.

Standalone: every path is resolved relative to this file, so the script can be
run from anywhere as ``python plot_dual_panel_table1.py``.

Outputs (in this script's own directory):
  - dual_panel_val_auc_table1.png   (300 dpi)
  - dual_panel.pdf                   (single-page PDF)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Configuration (paths resolved from this file so it is portable).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# training_curves/casgnet_vs_starnet/ -> training_curves/ -> excel_aligned/
DATA_DIR = SCRIPT_DIR.parent / "data"
TABLE1_DIR = SCRIPT_DIR.parents[1] / "table1_final_package"
OUT_DIR = SCRIPT_DIR

MODELS = ["casgnet", "starnet"]
DISPLAY_NAME = {"casgnet": "CasGNet", "starnet": "StarNet"}
COLORS = {"casgnet": "#1f77b4", "starnet": "#d62728"}

# StarNet checkpoint is named starnet_s1 in table1_final_package artefacts.
TABLE1_MODEL_NAME = {"casgnet": "casgnet", "starnet": "starnet_s1"}

# Manually supplied AUC values for the right-panel annotation (subset217, n=230).
# These match TABLE1_SUMMARY.csv repro_auc (the value reproduced from the
# checkpoint, which is the value reported in Table1).
TABLE1_FINAL_AUC = {"casgnet": 0.965, "starnet": 0.946}

LINEWIDTH = 2.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_history_csv(model: str) -> Tuple[List[int], List[float]]:
    """Load epoch + val_auc from training_curves/data/{model}_history.csv."""
    path = DATA_DIR / f"{model}_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing history CSV: {path}")
    epochs, aucs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            aucs.append(float(row["val_auc"]))
    return epochs, aucs


def best_val_auc(aucs: List[float], epochs: List[int]) -> Tuple[int, float]:
    i = max(range(len(aucs)), key=lambda i: aucs[i])
    return epochs[i], aucs[i]


def load_table1_repro_auc() -> Dict[str, float]:
    """Read repro_auc from TABLE1_SUMMARY.csv; fall back to TABLE1_FINAL_AUC."""
    path = TABLE1_DIR / "TABLE1_SUMMARY.csv"
    out: Dict[str, float] = {}
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                model = row["model"]
                if model in {TABLE1_MODEL_NAME["casgnet"], TABLE1_MODEL_NAME["starnet"]}:
                    # parse "0.965(0.946-0.980)" -> 0.965
                    val = float(row["repro_auc"].split("(")[0])
                    out[model] = val
    # normalise keys back to casgnet/starnet
    norm: Dict[str, float] = {}
    for k in ("casgnet", "starnet"):
        norm[k] = out.get(TABLE1_MODEL_NAME[k], TABLE1_FINAL_AUC[k])
    return norm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_left_panel(ax: plt.Axes) -> None:
    """val_auc curves over 200 epochs with best points annotated."""
    for model in MODELS:
        epochs, aucs = load_history_csv(model)
        ax.plot(
            epochs, aucs,
            label=DISPLAY_NAME[model], color=COLORS[model],
            linewidth=LINEWIDTH,
        )
        best_ep, best_auc = best_val_auc(aucs, epochs)
        ax.scatter(
            [best_ep], [best_auc], color=COLORS[model],
            s=80, zorder=5, edgecolors="white", linewidths=1.3,
        )
        # Offset the two labels so they don't collide.
        offset = (10, 14) if model == "casgnet" else (10, -22)
        ax.annotate(
            f"{DISPLAY_NAME[model]} best\n{best_auc:.4f} @ ep{best_ep}",
            xy=(best_ep, best_auc),
            xytext=offset, textcoords="offset points",
            color=COLORS[model], fontsize=9, ha="left",
            arrowprops=dict(arrowstyle="->", color=COLORS[model], lw=0.8),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC over training (200 epochs)")
    ax.set_ylim(0.88, 0.95)
    ax.set_xlim(1, 200)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)


def plot_right_panel(ax: plt.Axes, table1_auc: Dict[str, float]) -> None:
    """Table1 final macro-AUC bar chart with gap annotation."""
    names = [DISPLAY_NAME[m] for m in MODELS]
    values = [table1_auc[m] for m in MODELS]
    colors = [COLORS[m] for m in MODELS]

    bars = ax.bar(
        names, values, color=colors,
        width=0.5, edgecolor="black", linewidth=0.8,
    )
    # Annotate value on top of each bar.
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.002,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=bar.get_facecolor(),
        )

    gap = values[0] - values[1]
    # Draw a bracket between the two bar tops showing the gap.
    y_bracket = max(values) + 0.018
    ax.plot(
        [0, 0, 1, 1],
        [y_bracket - 0.004, y_bracket, y_bracket, y_bracket - 0.004],
        color="black", linewidth=1.0,
    )
    ax.text(
        0.5, y_bracket + 0.001,
        f"gap = {gap:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

    ax.set_ylabel("Macro AUC")
    ax.set_title("Table1 (subset217, n=230) Final AUC")
    ax.set_ylim(0.90, 1.00)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    # Colour the y-axis tick labels.
    ax.set_axisbelow(True)


def plot_dual_panel(table1_auc: Dict[str, float]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    plot_left_panel(axes[0])
    plot_right_panel(axes[1], table1_auc)
    fig.suptitle(
        "CasGNet vs StarNet — validation trajectory and Table1 final AUC",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Best val_auc (from training_curves/data/) ===")
    for model in MODELS:
        eps, aucs = load_history_csv(model)
        ep, auc = best_val_auc(aucs, eps)
        print(f"  {model:8s}: best={auc:.4f} @ ep{ep}  (n={len(aucs)} epochs)")

    table1_auc = load_table1_repro_auc()
    print("\n=== Table1 final macro-AUC (subset217, n=230) ===")
    for model in MODELS:
        print(f"  {DISPLAY_NAME[model]:8s}: {table1_auc[model]:.4f}")
    print(f"  gap = {table1_auc['casgnet'] - table1_auc['starnet']:+.4f}")

    fig = plot_dual_panel(table1_auc)

    png_path = OUT_DIR / "dual_panel_val_auc_table1.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\n[OK] saved {png_path}")

    pdf_path = OUT_DIR / "dual_panel.pdf"
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    print(f"[OK] saved {pdf_path}")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
