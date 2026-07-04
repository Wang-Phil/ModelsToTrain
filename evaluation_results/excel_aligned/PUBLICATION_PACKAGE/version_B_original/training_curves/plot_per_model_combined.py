"""Per-model combined plots (train_loss + val_auc on dual y-axis) and a
summary PDF combining all curves into one document for paper insertion.

Data source: checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json

Outputs (under evaluation_results/excel_aligned/training_curves/):
  - per_model_combined/{model}_loss_auc.png  : 8 dual-axis combined plots
  - training_curves_all.pdf                  : summary PDF (4 pages)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "old_data_supcon_compare_v3"
OUT_DIR = PROJECT_ROOT / "evaluation_results" / "excel_aligned" / "training_curves"
PER_MODEL_COMBINED_DIR = OUT_DIR / "per_model_combined"
PER_MODEL_DIR = OUT_DIR / "per_model"

MODELS = [
    "casgnet",
    "starnet",
    "densenet121",
    "resnet18",
    "resnet50",
    "mobilenetv4_m",
    "googlenet",
    "lsnet_b",
]

# Display name -> checkpoint dir suffix (without the `_ce_only` suffix).
MODEL_TO_CKPT = {
    "casgnet": "casgnet_s1",
    "starnet": "starnet_s1",
}

COLORS = {
    "casgnet": "#1f77b4",
    "starnet": "#ff7f0e",
    "densenet121": "#2ca02c",
    "resnet18": "#d62728",
    "resnet50": "#9467bd",
    "mobilenetv4_m": "#8c564b",
    "googlenet": "#e377c2",
    "lsnet_b": "#17becf",
}

LINEWIDTH = 1.8


def ckpt_name(model: str) -> str:
    return MODEL_TO_CKPT.get(model, model)


def load_history(model: str) -> dict:
    path = CKPT_DIR / f"{ckpt_name(model)}_ce_only" / "history.json"
    if not path.exists():
        print(f"[WARN] missing history.json for {model}: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def plot_combined(model: str, out_path: Path) -> None:
    h = load_history(model)
    if not h or "train_loss" not in h or "val_auc" not in h:
        print(f"[WARN] skip combined {model}")
        return
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(h["epoch"], h["train_loss"], color=COLORS[model],
             linewidth=LINEWIDTH, label="train_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color=COLORS[model])
    ax1.tick_params(axis="y", labelcolor=COLORS[model])
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(h["epoch"], h["val_auc"], color="#222222",
             linewidth=LINEWIDTH, linestyle="--", label="val_auc")
    ax2.set_ylabel("Validation AUC", color="#222222")
    ax2.tick_params(axis="y", labelcolor="#222222")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9)

    ax1.set_title(f"{model} - Training Loss & Validation AUC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def plot_overlay(field: str, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for model in MODELS:
        h = load_history(model)
        if not h or field not in h:
            continue
        ax.plot(h["epoch"], h[field], label=model,
                color=COLORS[model], linewidth=LINEWIDTH)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_per_model_single(model: str, field: str, ylabel: str) -> plt.Figure:
    h = load_history(model)
    fig, ax = plt.subplots(figsize=(8, 5))
    if h and field in h:
        ax.plot(h["epoch"], h[field], color=COLORS[model], linewidth=LINEWIDTH)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model} - {ylabel}")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def build_pdf(pdf_path: Path) -> None:
    with PdfPages(pdf_path) as pdf:
        # Page 1: summary loss
        fig = plot_overlay("train_loss", "Training Loss",
                           "Training Loss Curves (8 Main Models, CE-only)")
        pdf.savefig(fig); plt.close(fig)
        # Page 2: summary AUC
        fig = plot_overlay("val_auc", "Validation AUC",
                           "Validation AUC Curves (8 Main Models, CE-only)")
        pdf.savefig(fig); plt.close(fig)
        # Pages 3..10: per-model combined
        for model in MODELS:
            h = load_history(model)
            if not h:
                continue
            fig, ax1 = plt.subplots(figsize=(9, 5.5))
            ax1.plot(h["epoch"], h["train_loss"], color=COLORS[model],
                     linewidth=LINEWIDTH, label="train_loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss", color=COLORS[model])
            ax1.tick_params(axis="y", labelcolor=COLORS[model])
            ax1.grid(True, linestyle="--", alpha=0.4)
            ax2 = ax1.twinx()
            ax2.plot(h["epoch"], h["val_auc"], color="#222222",
                     linewidth=LINEWIDTH, linestyle="--", label="val_auc")
            ax2.set_ylabel("Validation AUC", color="#222222")
            ax2.tick_params(axis="y", labelcolor="#222222")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", framealpha=0.9)
            ax1.set_title(f"{model} - Training Loss & Validation AUC")
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)
    print(f"[OK] saved {pdf_path}")


def main() -> None:
    PER_MODEL_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        plot_combined(model, PER_MODEL_COMBINED_DIR / f"{model}_loss_auc.png")
    build_pdf(OUT_DIR / "training_curves_all.pdf")
    print("\nDone.")


if __name__ == "__main__":
    main()
