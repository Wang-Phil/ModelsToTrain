# Training Curves — 8 Main Models (CE-only)

This folder contains plotting data, scripts, and output figures for the training
curves of the 8 main models trained under the CE-only baseline
(`old_data_supcon_compare_v3`).

## Data Source

All curves are derived from:

```
/home/ln/wangweicheng/ModelsTotrain/checkpoints/old_data_supcon_compare_v3/{model}_ce_only/history.json
```

Each `history.json` is a JSON object with three parallel arrays of length 200
(200 epochs of CE-only training):

| key          | type    | description                       |
|--------------|---------|-----------------------------------|
| `epoch`      | int[]   | epoch index (1..200)              |
| `train_loss` | float[] | per-epoch training loss           |
| `val_auc`    | float[] | per-epoch validation AUC          |

> No `train_acc` / `val_acc` are logged in CE-only training, so curves are
> limited to `train_loss` and `val_auc` (val_auc is the closest proxy for
> validation accuracy).

## Models Included (8)

| # | model          |
|---|----------------|
| 1 | casgnet        |
| 2 | starnet        |
| 3 | densenet121    |
| 4 | resnet18       |
| 5 | resnet50       |
| 6 | mobilenetv4_m  |
| 7 | googlenet      |
| 8 | lsnet_b        |

## Files in this folder

### Plotting data (CSV)

| file                                                        | description                                            |
|-------------------------------------------------------------|--------------------------------------------------------|
| `training_curves_data.csv`                                  | combined long-form CSV: `model,epoch,train_loss,val_auc` (1600 rows = 8 × 200) |
| `data/{model}_history.csv`                                  | per-model CSV (8 files), same columns as above         |
| `data/best_val_auc_summary.csv`                             | summary: `model,best_epoch,best_val_auc,final_epoch,final_val_auc,final_train_loss` |

### Plotting scripts (Python, standalone)

All scripts auto-resolve the project root via `Path(__file__).resolve().parents[3]`,
so they can be run from any working directory. No hardcoded absolute paths.

| script                                  | description                                            | outputs |
|-----------------------------------------|--------------------------------------------------------|---------|
| `plot_8_models_training_curves.py`      | Overlay + per-model single-axis plots                  | `training_loss_curves.png`, `val_auc_curves.png`, `per_model/{model}_loss.png`, `per_model/{model}_val_auc.png` |
| `annotate_curves.py`                    | Overlay with final-epoch dots + value labels, best-epoch triangles | `annotated/training_loss_curves_annotated.png`, `annotated/val_auc_curves_annotated.png` |
| `plot_per_model_combined.py`            | Per-model dual-axis (loss + AUC) PNGs and a summary PDF | `per_model_combined/{model}_loss_auc.png` (8), `training_curves_all.pdf` |
| `plot_casgnet_vs_starnet.py`            | Paper-focused 2-model (CasGNet vs StarNet) comparison   | `casgnet_vs_starnet/casgnet_vs_starnet_{loss,val_auc,combined}.png`, `casgnet_vs_starnet/casgnet_vs_starnet.pdf` |
| `export_training_curves_csv.py`         | Exports the CSV data files                             | `training_curves_data.csv`, `data/{model}_history.csv`, `data/best_val_auc_summary.csv` |

### Output plots (PNG / PDF)

| folder / file                  | contents                                                     |
|--------------------------------|--------------------------------------------------------------|
| `training_loss_curves.png`     | 8-model overlaid training loss                               |
| `val_auc_curves.png`           | 8-model overlaid validation AUC                              |
| `annotated/`                   | annotated overlay versions (final + best epoch markers)      |
| `per_model/`                   | 16 PNGs: `{model}_loss.png` and `{model}_val_auc.png`        |
| `per_model_combined/`          | 8 PNGs: `{model}_loss_auc.png` (dual-axis)                   |
| `training_curves_all.pdf`      | 10-page PDF: 2 overlay + 8 per-model combined                |
| `casgnet_vs_starnet/`          | Paper-focused 2-model comparison (CasGNet vs StarNet only)   |

## `casgnet_vs_starnet/` — Paper-focused 2-model comparison

This subfolder contains a focused head-to-head comparison between **CasGNet**
and the **StarNet** baseline only, deliberately excluding the other six
baselines. The full 8-model overlay is useful as supplementary material, but
in the paper the CasGNet-vs-StarNet margin is small relative to the spread of
the other six models, so a 2-model plot keeps the visual story clean.

| file                                              | contents                                                       |
|---------------------------------------------------|----------------------------------------------------------------|
| `casgnet_vs_starnet_loss.png`                     | 2-line training loss overlay (CasGNet=blue, StarNet=red)       |
| `casgnet_vs_starnet_val_auc.png`                  | 2-line validation AUC overlay                                  |
| `casgnet_vs_starnet_combined.png`                 | Side-by-side loss + val_auc subplots                           |
| `casgnet_vs_starnet.pdf`                          | 3-page PDF: loss, val_auc, combined                            |

Generated by `plot_casgnet_vs_starnet.py` (same data source and `MODEL_TO_CKPT`
mapping as the 8-model scripts; standalone, no hardcoded paths).

## How to regenerate

All scripts require `matplotlib` only.

```bash
cd /home/ln/wangweicheng/ModelsTotrain

# 1. Export CSV data (idempotent)
python evaluation_results/excel_aligned/training_curves/export_training_curves_csv.py

# 2. Generate overlay + per-model plots
python evaluation_results/excel_aligned/training_curves/plot_8_models_training_curves.py

# 3. Generate annotated overlays
python evaluation_results/excel_aligned/training_curves/annotate_curves.py

# 4. Generate per-model dual-axis combined plots + summary PDF
python evaluation_results/excel_aligned/training_curves/plot_per_model_combined.py

# 5. Generate paper-focused 2-model (CasGNet vs StarNet) comparison
python evaluation_results/excel_aligned/training_curves/plot_casgnet_vs_starnet.py
```

## Best val_auc summary (from `data/best_val_auc_summary.csv`)

| model          | best_epoch | best_val_auc | final_val_auc | final_train_loss |
|----------------|-----------:|-------------:|--------------:|-----------------:|
| casgnet        | 157 | 0.9449 | 0.9202 | 0.1775 |
| starnet        | 198 | 0.9351 | 0.9071 | 0.2127 |
| densenet121    | 180 | 0.9330 | 0.9227 | 0.2198 |
| resnet18       |  81 | 0.9343 | 0.8910 | 0.2482 |
| resnet50       | 158 | 0.9221 | 0.9036 | 0.4805 |
| mobilenetv4_m  | 141 | 0.9364 | 0.9112 | 0.4757 |
| googlenet      | 143 | 0.9296 | 0.9151 | 0.4738 |
| lsnet_b        | 104 | 0.9215 | 0.9159 | 0.2153 |
