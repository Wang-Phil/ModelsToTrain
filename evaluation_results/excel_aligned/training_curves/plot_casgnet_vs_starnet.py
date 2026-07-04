"""Paper-focused 2-model training curve comparison: CasGNet vs StarNet.

This script generates a focused comparison between our CasGNet and the
StarNet baseline only (instead of the full 8-model overlay), so the paper
can highlight the CasGNet-vs-StarNet head-to-head without the visual clutter
of six additional baselines that finish close to CasGNet.

Data source: checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json
Each history.json contains: epoch, train_loss, val_auc (200 epochs).

Outputs (under evaluation_results/excel_aligned/training_curves/casgnet_vs_starnet/):
  - casgnet_vs_starnet_loss.png       : 2-line training loss overlay
  - casgnet_vs_starnet_val_auc.png    : 2-line validation AUC overlay
  - casgnet_vs_starnet_combined.png   : side-by-side loss + val_auc subplots
  - casgnet_vs_starnet.pdf            : 3-page PDF (loss, auc, combined)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "old_data_supcon_compare_v3"
OUT_DIR = (
    PROJECT_ROOT
    / "evaluation_results"
    / "excel_aligned"
    / "training_curves"
    / "casgnet_vs_starnet"
)

# Only the two models of interest for the paper-focused comparison.
MODELS = ["casgnet", "starnet"]

# Display name -> checkpoint dir suffix (without the `_ce_only` suffix).
# casgnet and starnet checkpoints are stored under casgnet_s1_ce_only and
# starnet_s1_ce_only; only the display/output name is shortened.
MODEL_TO_CKPT = {
    "casgnet": "casgnet_s1",
    "starnet": "starnet_s1",
}

# Display names for legend / titles.
DISPLAY_NAME = {
    "casgnet": "CasGNet",
    "starnet": "StarNet",
}

# Distinct colors: CasGNet=blue, StarNet=orange-red.
COLORS = {
    "casgnet": "#1f77b4",  # blue
    "starnet": "#d62728",  # red
}

LINEWIDTH = 2.2


def ckpt_name(model: str) -> str:
    return MODEL_TO_CKPT.get(model, model)


def load_history(model: str) -> dict:
    path = CKPT_DIR / f"{ckpt_name(model)}_ce_only" / "history.json"
    if not path.exists():
        print(f"[WARN] missing history.json for {model}: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def _plot_overlay(field: str, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in MODELS:
        h = load_history(model)
        if not h or field not in h:
            print(f"[WARN] field '{field}' missing for {model}")
            continue
        ax.plot(
            h["epoch"],
            h[field],
            label=DISPLAY_NAME[model],
            color=COLORS[model],
            linewidth=LINEWIDTH,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    return fig


def _plot_combined() -> plt.Figure:
    fig, (ax_loss, ax_auc) = plt.subplots(
        1, 2, figsize=(14, 5.5)
    )
    for model in MODELS:
        h = load_history(model)
        if not h:
            continue
        ax_loss.plot(
            h["epoch"],
            h["train_loss"],
            label=DISPLAY_NAME[model],
            color=COLORS[model],
            linewidth=LINEWIDTH,
        )
        ax_auc.plot(
            h["epoch"],
            h["val_auc"],
            label=DISPLAY_NAME[model],
            color=COLORS[model],
            linewidth=LINEWIDTH,
        )
    for ax, ylabel, title in (
        (ax_loss, "Training Loss", "Training Loss"),
        (ax_auc, "Validation AUC", "Validation AUC"),
    ):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", framealpha=0.9)
    fig.suptitle("CasGNet vs StarNet", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Field availability report
    print("=== Field availability (CasGNet vs StarNet) ===")
    for model in MODELS:
        h = load_history(model)
        n = len(h.get("epoch", []))
        print(f"  {model:10s}: epochs={n:3d}  fields={list(h.keys()) if h else []}")

    # 1. Training loss overlay
    fig = _plot_overlay(
        "train_loss",
        ylabel="Training Loss",
        title="Training Loss: CasGNet vs StarNet",
    )
    loss_png = OUT_DIR / "casgnet_vs_starnet_loss.png"
    fig.savefig(loss_png, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {loss_png}")

    # 2. Validation AUC overlay
    fig = _plot_overlay(
        "val_auc",
        ylabel="Validation AUC",
        title="Validation AUC: CasGNet vs StarNet",
    )
    auc_png = OUT_DIR / "casgnet_vs_starnet_val_auc.png"
    fig.savefig(auc_png, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {auc_png}")

    # 3. Combined side-by-side
    fig = _plot_combined()
    combined_png = OUT_DIR / "casgnet_vs_starnet_combined.png"
    fig.savefig(combined_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved {combined_png}")

    # 4. PDF (3 pages: loss, auc, combined)
    pdf_path = OUT_DIR / "casgnet_vs_starnet.pdf"
    with PdfPages(pdf_path) as pdf:
        for field, ylabel, title in (
            ("train_loss", "Training Loss", "Training Loss: CasGNet vs StarNet"),
            ("val_auc", "Validation AUC", "Validation AUC: CasGNet vs StarNet"),
        ):
            fig = _plot_overlay(field, ylabel=ylabel, title=title)
            pdf.savefig(fig)
            plt.close(fig)
        fig = _plot_combined()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(f"[OK] saved {pdf_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
