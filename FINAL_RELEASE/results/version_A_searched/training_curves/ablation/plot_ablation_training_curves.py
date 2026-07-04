#!/usr/bin/env python3
"""Plot training loss + val AUC curves for the 8 CasGNet/Starnet ablation variants.

Data source:
  checkpoints/starnetsk_sk_kernel_ablation/casgnet_s1_ab{000..111}_ce_only/history.json
Each history.json contains per-epoch: epoch, train_loss, val_auc (200 epochs).

Ablation bits (SA, GRN, SK-UNIT) → friendly name + display title:
  ab000 → starnet_baseline    → "StarNet (baseline)"
  ab001 → casgnet_only_skunit → "CasGNet only SK-UNIT"
  ab010 → casgnet_only_grn    → "CasGNet only GRN"
  ab011 → casgnet_no_sa       → "CasGNet w/o SA"
  ab100 → casgnet_only_sa     → "CasGNet only SA"
  ab101 → casgnet_no_grn      → "CasGNet w/o GRN"
  ab110 → casgnet_no_skunit   → "CasGNet w/o SK-UNIT"
  ab111 → casgnet_full        → "CasGNet (full)"

Outputs (under evaluation_results/excel_aligned/training_curves/ablation/):
  - ablation_training_curves_data.csv  : combined CSV (variant,variant_name,epoch,train_loss,val_auc)
  - data/{variant_name}_history.csv    : per-variant history CSV
  - ablation_curves_all.pdf            : multi-panel PDF (loss + AUC overlaid for all 8 variants)
  - loss_overlay.png                   : train_loss overlay with legend
  - auc_overlay.png                    : val_auc overlay with legend (best epoch annotated)
  - per_model/{variant_name}/loss_auc_combined.png : per-variant dual-axis plot
  - summary.png                        : 2x4 grid of per-variant loss+AUC combined
  - best_val_auc_summary.csv           : variant_name,ab_code,best_val_auc,best_epoch,final_val_auc,final_train_loss
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "starnetsk_sk_kernel_ablation"
OUT_DIR = HERE

# ab_code -> (friendly_name, display_title, is_highlight)
VARIANTS = [
    ("ab000", "starnet_baseline",    "StarNet (baseline)",      True),   # baseline highlight
    ("ab001", "casgnet_only_skunit", "CasGNet only SK-UNIT",    False),
    ("ab010", "casgnet_only_grn",    "CasGNet only GRN",        False),
    ("ab011", "casgnet_no_sa",       "CasGNet w/o SA",          False),
    ("ab100", "casgnet_only_sa",     "CasGNet only SA",         False),
    ("ab101", "casgnet_no_grn",      "CasGNet w/o GRN",         False),
    ("ab110", "casgnet_no_skunit",   "CasGNet w/o SK-UNIT",     False),
    ("ab111", "casgnet_full",        "CasGNet (full)",          True),   # full highlight
]

# tab10 colorblind-friendly palette
_TAB10 = plt.get_cmap("tab10").colors
COLORS = {
    "starnet_baseline":    _TAB10[1],  # orange  — baseline
    "casgnet_only_skunit": _TAB10[2],  # green
    "casgnet_only_grn":    _TAB10[3],  # red
    "casgnet_no_sa":       _TAB10[4],  # purple
    "casgnet_only_sa":     _TAB10[5],  # brown
    "casgnet_no_grn":      _TAB10[6],  # pink
    "casgnet_no_skunit":   _TAB10[7],  # gray
    "casgnet_full":        _TAB10[0],  # blue   — full
}


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
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


def history_path(ab_code: str) -> Path:
    return CKPT_DIR / f"casgnet_s1_{ab_code}_ce_only" / "history.json"


def load_history(ab_code: str) -> dict:
    p = history_path(ab_code)
    if not p.is_file():
        print(f"[WARN] missing {p}")
        return {}
    with open(p) as f:
        return json.load(f)


def to_rows(h: dict, ab_code: str, variant_name: str) -> list[dict]:
    n = len(h.get("epoch", []))
    rows = []
    for i in range(n):
        rows.append(
            {
                "variant": ab_code,
                "variant_name": variant_name,
                "epoch": h["epoch"][i],
                "train_loss": h["train_loss"][i],
                "val_auc": h["val_auc"][i],
            }
        )
    return rows


def best_auc_idx(val_auc: list[float]) -> int:
    arr = np.asarray(val_auc, dtype=float)
    if arr.size == 0:
        return -1
    # ignore initial 0.0 placeholder if present at index 0 only
    return int(np.argmax(arr))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_overlay(
    histories: dict, field: str, ylabel: str, title: str, out_path: Path,
    annotate_best: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for ab_code, vname, vtitle, highlight in VARIANTS:
        h = histories.get(ab_code)
        if not h or field not in h:
            continue
        lw = 3.0 if highlight else 1.6
        alpha = 1.0 if highlight else 0.85
        ax.plot(
            h["epoch"],
            h[field],
            label=vtitle,
            color=COLORS[vname],
            linewidth=lw,
            alpha=alpha,
            zorder=5 if highlight else 3,
        )
        if annotate_best and field == "val_auc":
            bi = best_auc_idx(h["val_auc"])
            if bi >= 0:
                ax.scatter(
                    [h["epoch"][bi]], [h["val_auc"][bi]],
                    color=COLORS[vname], s=42, zorder=6,
                    edgecolor="black", linewidth=0.5,
                )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def plot_per_variant_combined(
    h: dict, vtitle: str, color, out_path: Path,
) -> None:
    if not h:
        return
    epochs = h["epoch"]
    loss = h["train_loss"]
    auc = h["val_auc"]

    fig, ax_loss = plt.subplots(figsize=(8, 5))
    ax_auc = ax_loss.twinx()
    ax_auc.grid(False)

    l1, = ax_loss.plot(epochs, loss, color=color, linewidth=1.9, label="Train loss")
    l2, = ax_auc.plot(epochs, auc, color="#444444", linewidth=1.6, linestyle="--", label="Val AUC")

    bi = best_auc_idx(auc)
    if bi >= 0:
        ax_auc.scatter([epochs[bi]], [auc[bi]], color="#cc2222", s=55,
                       edgecolor="black", linewidth=0.5, zorder=6,
                       label=f"Best AUC (ep {epochs[bi]})")
        ax_auc.annotate(
            f"ep {epochs[bi]}\nAUC={auc[bi]:.3f}",
            xy=(epochs[bi], auc[bi]),
            xytext=(0.62, 0.05), textcoords="axes fraction",
            fontsize=8, color="#cc2222",
            arrowprops=dict(arrowstyle="->", color="#cc2222", lw=0.8),
        )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss", color=color)
    ax_loss.tick_params(axis="y", labelcolor=color)
    ax_auc.set_ylabel("Validation AUC", color="#444444")
    ax_auc.tick_params(axis="y", labelcolor="#444444")
    ax_loss.set_title(vtitle)
    ax_loss.grid(True, linestyle="--", alpha=0.35)

    lines = [l1, l2]
    if bi >= 0:
        lines.append(ax_auc.collections[-1])  # the scatter
    ax_loss.legend(lines, [l.get_label() for l in lines], loc="best", framealpha=0.9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def plot_summary_grid(histories: dict, out_path: Path) -> None:
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 9))
    for idx, (ab_code, vname, vtitle, _) in enumerate(VARIANTS):
        r, c = divmod(idx, ncols)
        ax_loss = axes[r, c]
        ax_auc = ax_loss.twinx()
        ax_auc.grid(False)
        h = histories.get(ab_code)
        if not h:
            ax_loss.text(0.5, 0.5, "Missing", ha="center", va="center", transform=ax_loss.transAxes)
            ax_loss.set_title(vtitle)
            continue
        epochs = h["epoch"]
        loss = h["train_loss"]
        auc = h["val_auc"]
        color = COLORS[vname]
        ax_loss.plot(epochs, loss, color=color, linewidth=1.6, label="Train loss")
        ax_auc.plot(epochs, auc, color="#444444", linewidth=1.3, linestyle="--", label="Val AUC")
        bi = best_auc_idx(auc)
        if bi >= 0:
            ax_auc.scatter([epochs[bi]], [auc[bi]], color="#cc2222", s=30,
                           edgecolor="black", linewidth=0.4, zorder=6)
        ax_loss.set_title(vtitle, fontsize=11)
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("Loss", color=color, fontsize=9)
        ax_loss.tick_params(axis="y", labelcolor=color, labelsize=8)
        ax_auc.set_ylabel("AUC", color="#444444", fontsize=9)
        ax_auc.tick_params(axis="y", labelcolor="#444444", labelsize=8)
        ax_loss.tick_params(axis="x", labelsize=8)
        ax_loss.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("Ablation Variants — Training Loss & Validation AUC",
                 fontsize=14, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


def write_pdf(
    histories: dict, out_path: Path,
) -> None:
    """Multi-panel PDF: page 1 = loss overlay; page 2 = AUC overlay; per-variant pages."""
    with PdfPages(out_path) as pdf:
        # Page 1: loss overlay
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for ab_code, vname, vtitle, highlight in VARIANTS:
            h = histories.get(ab_code)
            if not h:
                continue
            ax.plot(h["epoch"], h["train_loss"], label=vtitle,
                    color=COLORS[vname],
                    linewidth=3.0 if highlight else 1.6,
                    alpha=1.0 if highlight else 0.85)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Training loss")
        ax.set_title("Ablation — Training Loss (all variants)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", ncol=2, framealpha=0.9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Page 2: AUC overlay
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for ab_code, vname, vtitle, highlight in VARIANTS:
            h = histories.get(ab_code)
            if not h:
                continue
            ax.plot(h["epoch"], h["val_auc"], label=vtitle,
                    color=COLORS[vname],
                    linewidth=3.0 if highlight else 1.6,
                    alpha=1.0 if highlight else 0.85)
            bi = best_auc_idx(h["val_auc"])
            if bi >= 0:
                ax.scatter([h["epoch"][bi]], [h["val_auc"][bi]],
                           color=COLORS[vname], s=42, edgecolor="black", linewidth=0.5, zorder=6)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC")
        ax.set_title("Ablation — Validation AUC (all variants)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", ncol=2, framealpha=0.9)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # Per-variant pages
        for ab_code, vname, vtitle, _ in VARIANTS:
            h = histories.get(ab_code)
            if not h:
                continue
            fig, ax_loss = plt.subplots(figsize=(8.5, 5.5))
            ax_auc = ax_loss.twinx(); ax_auc.grid(False)
            epochs, loss, auc = h["epoch"], h["train_loss"], h["val_auc"]
            color = COLORS[vname]
            ax_loss.plot(epochs, loss, color=color, linewidth=1.9, label="Train loss")
            ax_auc.plot(epochs, auc, color="#444444", linewidth=1.6, linestyle="--", label="Val AUC")
            bi = best_auc_idx(auc)
            if bi >= 0:
                ax_auc.scatter([epochs[bi]], [auc[bi]], color="#cc2222", s=55,
                               edgecolor="black", linewidth=0.5, zorder=6,
                               label=f"Best AUC (ep {epochs[bi]})")
                ax_auc.annotate(f"ep {epochs[bi]}\nAUC={auc[bi]:.3f}",
                                xy=(epochs[bi], auc[bi]),
                                xytext=(0.62, 0.05), textcoords="axes fraction",
                                fontsize=8, color="#cc2222",
                                arrowprops=dict(arrowstyle="->", color="#cc2222", lw=0.8))
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Training loss", color=color)
            ax_loss.tick_params(axis="y", labelcolor=color)
            ax_auc.set_ylabel("Validation AUC", color="#444444")
            ax_auc.tick_params(axis="y", labelcolor="#444444")
            ax_loss.set_title(vtitle)
            ax_loss.grid(True, linestyle="--", alpha=0.35)
            ax_loss.legend(loc="best", framealpha=0.9)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"[OK] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "per_model").mkdir(parents=True, exist_ok=True)

    histories: dict[str, dict] = {}
    all_rows: list[dict] = []
    best_rows: list[dict] = []

    for ab_code, vname, vtitle, _ in VARIANTS:
        h = load_history(ab_code)
        histories[ab_code] = h
        if not h:
            continue
        rows = to_rows(h, ab_code, vname)
        all_rows.extend(rows)

        # per-variant CSV
        per_csv = OUT_DIR / "data" / f"{vname}_history.csv"
        with open(per_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["variant", "variant_name", "epoch", "train_loss", "val_auc"])
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {per_csv}")

        # best / final
        bi = best_auc_idx(h["val_auc"])
        best_rows.append({
            "variant_name": vname,
            "ab_code": ab_code,
            "best_val_auc": h["val_auc"][bi] if bi >= 0 else "",
            "best_epoch": h["epoch"][bi] if bi >= 0 else "",
            "final_val_auc": h["val_auc"][-1],
            "final_train_loss": h["train_loss"][-1],
        })

    # combined CSV
    combined_csv = OUT_DIR / "ablation_training_curves_data.csv"
    with open(combined_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "variant_name", "epoch", "train_loss", "val_auc"])
        w.writeheader()
        w.writerows(all_rows)
    print(f"[OK] {combined_csv}")

    # best_val_auc_summary.csv
    best_csv = OUT_DIR / "best_val_auc_summary.csv"
    with open(best_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant_name", "ab_code", "best_val_auc",
                                          "best_epoch", "final_val_auc", "final_train_loss"])
        w.writeheader()
        w.writerows(best_rows)
    print(f"[OK] {best_csv}")

    apply_plot_style()

    # Overlays
    plot_overlay(histories, "train_loss", "Training loss",
                 "Ablation — Training Loss (8 variants)",
                 OUT_DIR / "loss_overlay.png", annotate_best=False)
    plot_overlay(histories, "val_auc", "Validation AUC",
                 "Ablation — Validation AUC (8 variants, ● = best epoch)",
                 OUT_DIR / "auc_overlay.png", annotate_best=True)

    # Per-variant combined
    for ab_code, vname, vtitle, _ in VARIANTS:
        h = histories.get(ab_code)
        if not h:
            continue
        per_dir = OUT_DIR / "per_model" / vname
        per_dir.mkdir(parents=True, exist_ok=True)
        plot_per_variant_combined(h, vtitle, COLORS[vname],
                                  per_dir / "loss_auc_combined.png")

    # Summary grid (2x4)
    plot_summary_grid(histories, OUT_DIR / "summary.png")

    # PDF
    write_pdf(histories, OUT_DIR / "ablation_curves_all.pdf")

    # Console summary
    print("\n=== best_val_auc_summary ===")
    df = pd.DataFrame(best_rows).sort_values("best_val_auc", ascending=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
