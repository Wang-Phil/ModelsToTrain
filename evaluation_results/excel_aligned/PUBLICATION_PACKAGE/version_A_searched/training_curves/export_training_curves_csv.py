"""Export training-curve plotting data to CSV files for downstream use.

Data source: checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json
Each history.json contains: epoch, train_loss, val_auc (200 epochs).

Outputs (under evaluation_results/excel_aligned/training_curves/):
  - training_curves_data.csv            : combined long-form (all 8 models)
  - data/{model}_history.csv            : one CSV per model (8 files)
  - data/best_val_auc_summary.csv       : best/final epoch summary for all 8
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CKPT_DIR = PROJECT_ROOT / "checkpoints" / "old_data_supcon_compare_v3"
OUT_DIR = PROJECT_ROOT / "evaluation_results" / "excel_aligned" / "training_curves"
DATA_DIR = OUT_DIR / "data"

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


def ckpt_name(model: str) -> str:
    return MODEL_TO_CKPT.get(model, model)


def load_history(model: str) -> dict:
    path = CKPT_DIR / f"{ckpt_name(model)}_ce_only" / "history.json"
    if not path.exists():
        print(f"[WARN] missing history.json for {model}: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    combined_rows = []
    summary_rows = []

    for model in MODELS:
        h = load_history(model)
        if not h:
            continue

        epochs = h["epoch"]
        train_losses = h["train_loss"]
        val_aucs = h["val_auc"]

        # Per-model CSV
        per_model_path = DATA_DIR / f"{model}_history.csv"
        with open(per_model_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "epoch", "train_loss", "val_auc"])
            for ep, tl, va in zip(epochs, train_losses, val_aucs):
                writer.writerow([model, ep, tl, va])
                combined_rows.append([model, ep, tl, va])
        print(f"[OK] saved {per_model_path}")

        # Best/final epoch summary
        best_idx = max(range(len(val_aucs)), key=lambda i: val_aucs[i])
        summary_rows.append([
            model,
            epochs[best_idx],       # best_epoch
            val_aucs[best_idx],     # best_val_auc
            epochs[-1],             # final_epoch
            val_aucs[-1],           # final_val_auc
            train_losses[-1],       # final_train_loss
        ])

    # Combined CSV (long form)
    combined_path = OUT_DIR / "training_curves_data.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "epoch", "train_loss", "val_auc"])
        writer.writerows(combined_rows)
    print(f"[OK] saved {combined_path}  ({len(combined_rows)} rows)")

    # Best val AUC summary
    summary_path = DATA_DIR / "best_val_auc_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "best_epoch", "best_val_auc",
                         "final_epoch", "final_val_auc", "final_train_loss"])
        writer.writerows(summary_rows)
    print(f"[OK] saved {summary_path}  ({len(summary_rows)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
