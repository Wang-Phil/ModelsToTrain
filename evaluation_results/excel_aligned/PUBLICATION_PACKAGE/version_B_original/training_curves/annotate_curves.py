"""Annotate the existing 8-model training curves with the final-epoch val_auc
for each model, so the user can see the alignment between curves and the
final Table1/Table2 produced data.

Reads from: checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json
Writes to:  evaluation_results/excel_aligned/training_curves/annotated/

Outputs:
  - training_loss_curves_annotated.png
  - val_auc_curves_annotated.png  (with final-epoch markers + value labels)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "old_data_supcon_compare_v3"
OUT_DIR = PROJECT_ROOT / "evaluation_results" / "excel_aligned" / "training_curves" / "annotated"

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
    with open(CKPT_DIR / f"{ckpt_name(model)}_ce_only" / "history.json") as f:
        return json.load(f)


def plot_val_auc_annotated() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    final_points = []
    for model in MODELS:
        h = load_history(model)
        ax.plot(h["epoch"], h["val_auc"], label=model, color=COLORS[model], linewidth=LINEWIDTH)
        last_epoch = h["epoch"][-1]
        last_val_auc = h["val_auc"][-1]
        final_points.append((model, last_epoch, last_val_auc))

    # Sort final points by val_auc descending so labels stack neatly
    final_points_sorted = sorted(final_points, key=lambda x: x[2], reverse=True)

    # Mark final epoch with a dot + label
    y_offsets = {}
    # Distribute labels vertically to avoid overlap
    sorted_y = [p[2] for p in final_points_sorted]
    for (model, ep, auc), y in zip(final_points_sorted, sorted_y):
        ax.scatter([ep], [auc], color=COLORS[model], s=55, zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"{model}\nfinal={auc:.4f}",
            xy=(ep, auc),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[model],
            fontweight="bold",
            va="center",
            ha="left",
        )

    # Also mark best val_auc with a small triangle on the curve
    for model in MODELS:
        h = load_history(model)
        best_idx = max(range(len(h["val_auc"])), key=lambda i: h["val_auc"][i])
        best_ep = h["epoch"][best_idx]
        best_auc = h["val_auc"][best_idx]
        ax.scatter([best_ep], [best_auc], marker="^", color=COLORS[model], s=40, zorder=4, edgecolors="black", linewidths=0.5, alpha=0.85)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Validation AUC Curves (8 Main Models, CE-only)\n● = final epoch  ▲ = best epoch (saved checkpoint)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", ncol=2, framealpha=0.9)
    # Leave room on right for labels
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(OUT_DIR / "val_auc_curves_annotated.png", dpi=200)
    plt.close(fig)
    print(f"[OK] saved {OUT_DIR / 'val_auc_curves_annotated.png'}")


def plot_train_loss_annotated() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.5))
    for model in MODELS:
        h = load_history(model)
        ax.plot(h["epoch"], h["train_loss"], label=model, color=COLORS[model], linewidth=LINEWIDTH)
        last_epoch = h["epoch"][-1]
        last_loss = h["train_loss"][-1]
        ax.scatter([last_epoch], [last_loss], color=COLORS[model], s=55, zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(
            f"{model}\nfinal={last_loss:.4f}",
            xy=(last_epoch, last_loss),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=8,
            color=COLORS[model],
            fontweight="bold",
            va="center",
            ha="left",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves (8 Main Models, CE-only)\n● = final epoch")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)
    fig.savefig(OUT_DIR / "training_loss_curves_annotated.png", dpi=200)
    plt.close(fig)
    print(f"[OK] saved {OUT_DIR / 'training_loss_curves_annotated.png'}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_train_loss_annotated()
    plot_val_auc_annotated()
    print("\nDone.")


if __name__ == "__main__":
    main()
