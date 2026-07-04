"""Plot training loss and validation AUC curves for the 8 main models.

Data source: checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json
Each history.json contains only: epoch, train_loss, val_auc (200 epochs).
No train_acc / val_acc are logged in CE-only training, so we plot train_loss
and val_auc only. val_auc is the closest proxy for validation accuracy.

Outputs (under evaluation_results/excel_aligned/training_curves/):
  - training_loss_curves.png        : 8-model overlaid training loss
  - val_auc_curves.png              : 8-model overlaid validation AUC
  - per_model/{model}_loss.png      : per-model training loss
  - per_model/{model}_val_auc.png   : per-model validation AUC
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "old_data_supcon_compare_v3"
OUT_DIR = PROJECT_ROOT / "evaluation_results" / "excel_aligned" / "training_curves"
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
# casgnet and starnet checkpoints are still stored under casgnet_s1_ce_only
# and starnet_s1_ce_only; only the display/output name is shortened.
MODEL_TO_CKPT = {
    "casgnet": "casgnet_s1",
    "starnet": "starnet_s1",
}

# Distinct, colorblind-friendly palette (tab10 subset reordered for contrast)
COLORS = {
    "casgnet": "#1f77b4",  # blue
    "starnet": "#ff7f0e",  # orange
    "densenet121": "#2ca02c",  # green
    "resnet18": "#d62728",  # red
    "resnet50": "#9467bd",  # purple
    "mobilenetv4_m": "#8c564b",  # brown
    "googlenet": "#e377c2",  # pink
    "lsnet_b": "#17becf",  # cyan
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


def plot_overlay(field: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for model in MODELS:
        h = load_history(model)
        if not h or field not in h:
            print(f"[WARN] field '{field}' missing for {model}")
            continue
        ax.plot(
            h["epoch"],
            h[field],
            label=model,
            color=COLORS[model],
            linewidth=LINEWIDTH,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def plot_per_model(model: str, field: str, ylabel: str, title: str, out_path: Path) -> None:
    h = load_history(model)
    if not h or field not in h:
        print(f"[WARN] skip per-model {model} {field}")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h["epoch"], h[field], color=COLORS[model], linewidth=LINEWIDTH)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] saved {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Field availability report
    print("=== Field availability ===")
    fields_seen = set()
    for model in MODELS:
        h = load_history(model)
        keys = list(h.keys()) if h else []
        fields_seen.update(keys)
        n = len(h.get("epoch", []))
        print(f"  {model:15s}: epochs={n:3d}  fields={keys}")
    print(f"Union of fields across all models: {sorted(fields_seen)}")

    has_train_acc = all("train_acc" in load_history(m) for m in MODELS)
    has_val_acc = all("val_acc" in load_history(m) for m in MODELS)
    print(f"train_acc available for all: {has_train_acc}")
    print(f"val_acc   available for all: {has_val_acc}")

    # Overlay curves
    plot_overlay(
        "train_loss",
        ylabel="Training Loss",
        title="Training Loss Curves (8 Main Models, CE-only)",
        out_path=OUT_DIR / "training_loss_curves.png",
    )
    plot_overlay(
        "val_auc",
        ylabel="Validation AUC",
        title="Validation AUC Curves (8 Main Models, CE-only)",
        out_path=OUT_DIR / "val_auc_curves.png",
    )

    # Per-model curves
    for model in MODELS:
        plot_per_model(
            model,
            "train_loss",
            ylabel="Training Loss",
            title=f"{model} - Training Loss",
            out_path=PER_MODEL_DIR / f"{model}_loss.png",
        )
        plot_per_model(
            model,
            "val_auc",
            ylabel="Validation AUC",
            title=f"{model} - Validation AUC",
            out_path=PER_MODEL_DIR / f"{model}_val_auc.png",
        )

    # Optional: train_acc overlay if present
    if has_train_acc:
        plot_overlay(
            "train_acc",
            ylabel="Training Accuracy",
            title="Training Accuracy Curves (8 Main Models)",
            out_path=OUT_DIR / "train_acc_curves.png",
        )
    if has_val_acc:
        plot_overlay(
            "val_acc",
            ylabel="Validation Accuracy",
            title="Validation Accuracy Curves (8 Main Models)",
            out_path=OUT_DIR / "val_acc_curves.png",
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
