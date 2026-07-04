"""Enhanced CasGNet vs StarNet visualizations to expose the small AUC gap.

The standard val_auc overlay looks indistinguishable after epoch ~50 because
both models stabilize in the 0.88-0.94 band and the gap is only ~0.01. This
script generates a set of alternative views (zoomed axes, per-epoch
difference, smoothed curves, last-100 zoom, filled-between, plus a combined
2x2 multi-panel) that magnify the difference for paper presentation.

Data source: checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json
Each history.json contains: epoch, train_loss, val_auc (200 epochs).

Outputs (in this script's own directory):
  - val_auc_zoomed.png         : y-axis [0.90, 0.95]
  - auc_difference.png         : per-epoch CasGNet - StarNet with +/- shading
  - val_auc_smoothed.png       : 10-epoch moving average (raw faint + smoothed bold)
  - val_auc_last100.png        : x-axis [100, 200], y-axis [0.90, 0.95]
  - val_auc_filled.png         : fill_between where CasGNet > StarNet
  - casgnet_vs_starnet_enhanced.png : 2x2 multi-panel (loss / auc / zoomed / diff)
  - casgnet_vs_starnet_enhanced.pdf : 7-page PDF (the 6 above + multi-panel)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Configuration (paths resolved from this file so it is portable).
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# training_curves/casgnet_vs_starnet/ -> training_curves/ -> excel_aligned/ -> evaluation_results/ -> PROJECT_ROOT
PROJECT_ROOT = SCRIPT_DIR.parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "old_data_supcon_compare_v3"
OUT_DIR = SCRIPT_DIR

MODELS = ["casgnet", "starnet"]

# Display name -> checkpoint dir suffix (without the `_ce_only` suffix).
MODEL_TO_CKPT = {
    "casgnet": "casgnet_s1",
    "starnet": "starnet_s1",
}

DISPLAY_NAME = {
    "casgnet": "CasGNet",
    "starnet": "StarNet",
}

COLORS = {
    "casgnet": "#1f77b4",  # blue
    "starnet": "#d62728",  # red
}

LINEWIDTH = 2.2
SMOOTH_WINDOW = 10  # epochs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def ckpt_name(model: str) -> str:
    return MODEL_TO_CKPT.get(model, model)


def load_history(model: str) -> dict:
    path = CKPT_DIR / f"{ckpt_name(model)}_ce_only" / "history.json"
    if not path.exists():
        print(f"[WARN] missing history.json for {model}: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def moving_average(values, window: int = SMOOTH_WINDOW):
    """Centered-ish moving average; uses 'same' length via cumulative trick.

    Uses a trailing/edge-aware window so the first/last few points remain
    meaningful (no NaN trimming). At index i, the window covers
    [max(0, i - window + 1), i].
    """
    out = []
    cum = 0.0
    for i, v in enumerate(values):
        cum += v
        if i >= window:
            cum -= values[i - window]
        denom = min(i + 1, window)
        out.append(cum / denom)
    return out


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------
def plot_zoomed() -> plt.Figure:
    """1. val_auc with y-axis [0.90, 0.95] and 0.01 gridlines."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        h = load_history(model)
        if not h:
            continue
        ax.plot(
            h["epoch"], h["val_auc"],
            label=DISPLAY_NAME[model], color=COLORS[model],
            linewidth=LINEWIDTH,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC (zoomed): CasGNet vs StarNet")
    ax.set_ylim(0.90, 0.95)
    ax.set_yticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    ax.grid(True, linestyle="--", alpha=0.5, which="both")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_difference() -> plt.Figure:
    """2. Per-epoch (CasGNet - StarNet) val_auc with green/red shading."""
    hc = load_history("casgnet")
    hs = load_history("starnet")
    if not (hc and hs):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.text(0.5, 0.5, "missing data", ha="center", va="center")
        return fig

    epochs = hc["epoch"]
    diff = [c - s for c, s in zip(hc["val_auc"], hs["val_auc"])]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, diff, color="#2c3e50", linewidth=1.6, zorder=3)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", zorder=2)

    # Shade positive (green) and negative (red) regions.
    ax.fill_between(
        epochs, diff, 0,
        where=[d >= 0 for d in diff],
        interpolate=True,
        color="#2ca02c", alpha=0.35, label="CasGNet > StarNet",
    )
    ax.fill_between(
        epochs, diff, 0,
        where=[d < 0 for d in diff],
        interpolate=True,
        color="#d62728", alpha=0.35, label="CasGNet < StarNet",
    )

    ymax = max(abs(min(diff)), abs(max(diff))) * 1.15
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CasGNet AUC - StarNet AUC")
    ax.set_title("CasGNet AUC advantage over StarNet (per epoch)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_smoothed() -> plt.Figure:
    """3. 10-epoch moving average; raw faint + smoothed bold; y [0.88, 0.95]."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        h = load_history(model)
        if not h:
            continue
        epochs = h["epoch"]
        raw = h["val_auc"]
        smoothed = moving_average(raw, SMOOTH_WINDOW)
        ax.plot(epochs, raw, color=COLORS[model], linewidth=1.0, alpha=0.3)
        ax.plot(
            epochs, smoothed,
            label=f"{DISPLAY_NAME[model]} (10-ep MA)",
            color=COLORS[model], linewidth=LINEWIDTH,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC (10-epoch moving average)")
    ax.set_ylim(0.88, 0.95)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_last100() -> plt.Figure:
    """4. Last 100 epochs zoom; annotate best val_auc per model."""
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for model in MODELS:
        h = load_history(model)
        if not h:
            continue
        epochs = h["epoch"]
        aucs = h["val_auc"]
        ax.plot(
            epochs, aucs,
            label=DISPLAY_NAME[model], color=COLORS[model],
            linewidth=LINEWIDTH,
        )
        # best in full history, but only annotate if within last 100 epochs
        best_i = max(range(len(aucs)), key=lambda i: aucs[i])
        best_epoch = epochs[best_i]
        best_auc = aucs[best_i]
        if best_epoch >= 100:
            ax.scatter([best_epoch], [best_auc], color=COLORS[model],
                       s=70, zorder=5, edgecolors="white", linewidths=1.2)
            ax.annotate(
                f"{DISPLAY_NAME[model]} best: {best_auc:.4f} @ ep{best_epoch}",
                xy=(best_epoch, best_auc),
                xytext=(8, 10), textcoords="offset points",
                color=COLORS[model], fontsize=9,
                arrowprops=dict(arrowstyle="->", color=COLORS[model], lw=0.8),
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC (last 100 epochs, zoomed)")
    ax.set_xlim(100, 200)
    ax.set_ylim(0.90, 0.95)
    ax.set_yticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_filled() -> plt.Figure:
    """5. Standard val_auc overlay with fill_between where CasGNet > StarNet."""
    hc = load_history("casgnet")
    hs = load_history("starnet")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    if not (hc and hs):
        ax.text(0.5, 0.5, "missing data", ha="center", va="center")
        return fig
    epochs = hc["epoch"]
    ca = hc["val_auc"]
    sa = hs["val_auc"]
    ax.plot(epochs, ca, label=DISPLAY_NAME["casgnet"],
            color=COLORS["casgnet"], linewidth=LINEWIDTH)
    ax.plot(epochs, sa, label=DISPLAY_NAME["starnet"],
            color=COLORS["starnet"], linewidth=LINEWIDTH)
    ax.fill_between(
        epochs, ca, sa,
        where=[c >= s for c, s in zip(ca, sa)],
        interpolate=True,
        color="#9ecae1", alpha=0.5,
        label="CasGNet > StarNet",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC: CasGNet vs StarNet (gap shaded)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_multipanel() -> plt.Figure:
    """6. 2x2 combined multi-panel for paper insertion."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: full training loss
    ax = axes[0, 0]
    for model in MODELS:
        h = load_history(model)
        if not h:
            continue
        ax.plot(h["epoch"], h["train_loss"],
                label=DISPLAY_NAME[model], color=COLORS[model],
                linewidth=LINEWIDTH)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)

    # Top-right: full val_auc
    ax = axes[0, 1]
    hc, hs = load_history("casgnet"), load_history("starnet")
    for model, h in (("casgnet", hc), ("starnet", hs)):
        if not h:
            continue
        ax.plot(h["epoch"], h["val_auc"],
                label=DISPLAY_NAME[model], color=COLORS[model],
                linewidth=LINEWIDTH)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC (full)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)

    # Bottom-left: zoomed val_auc [0.90, 0.95]
    ax = axes[1, 0]
    for model, h in (("casgnet", hc), ("starnet", hs)):
        if not h:
            continue
        ax.plot(h["epoch"], h["val_auc"],
                label=DISPLAY_NAME[model], color=COLORS[model],
                linewidth=LINEWIDTH)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC (zoomed [0.90, 0.95])")
    ax.set_ylim(0.90, 0.95)
    ax.set_yticks([0.90, 0.91, 0.92, 0.93, 0.94, 0.95])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", framealpha=0.9)

    # Bottom-right: difference curve
    ax = axes[1, 1]
    if hc and hs:
        epochs = hc["epoch"]
        diff = [c - s for c, s in zip(hc["val_auc"], hs["val_auc"])]
        ax.plot(epochs, diff, color="#2c3e50", linewidth=1.6, zorder=3)
        ax.axhline(0, color="black", linewidth=1.0, zorder=2)
        ax.fill_between(epochs, diff, 0,
                        where=[d >= 0 for d in diff], interpolate=True,
                        color="#2ca02c", alpha=0.35, label="CasGNet > StarNet")
        ax.fill_between(epochs, diff, 0,
                        where=[d < 0 for d in diff], interpolate=True,
                        color="#d62728", alpha=0.35, label="CasGNet < StarNet")
        ymax = max(abs(min(diff)), abs(max(diff))) * 1.15
        ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CasGNet - StarNet")
    ax.set_title("CasGNet AUC advantage (per epoch)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)

    fig.suptitle("CasGNet vs StarNet — enhanced comparison", fontsize=14, y=1.00)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Field availability (CasGNet vs StarNet) ===")
    for model in MODELS:
        h = load_history(model)
        n = len(h.get("epoch", []))
        print(f"  {model:10s}: epochs={n:3d}  fields={list(h.keys()) if h else []}")

    figures = [
        ("val_auc_zoomed.png", plot_zoomed),
        ("auc_difference.png", plot_difference),
        ("val_auc_smoothed.png", plot_smoothed),
        ("val_auc_last100.png", plot_last100),
        ("val_auc_filled.png", plot_filled),
    ]

    saved_pngs = []
    for name, fn in figures:
        fig = fn()
        path = OUT_DIR / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved_pngs.append(path)
        print(f"[OK] saved {path}")

    # 6. Combined multi-panel PNG.
    fig = plot_multipanel()
    multi_png = OUT_DIR / "casgnet_vs_starnet_enhanced.png"
    fig.savefig(multi_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {multi_png}")

    # 7. PDF: 6 single figures + multi-panel = 7 pages.
    pdf_path = OUT_DIR / "casgnet_vs_starnet_enhanced.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, fn in figures:
            fig = fn()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        fig = plot_multipanel()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"[OK] saved {pdf_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
