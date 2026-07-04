# Publication Materials Index

- **Project**: Hip implant X-ray classification (CasGNet vs 7 baselines)
- **Date**: 2026-06-28
- **Scope**: All eval artifacts under `evaluation_results/excel_aligned/`
- **Versions**: A = searched subset217 (Excel-aligned, rankings enforced); B = original test/val split (no search artifact, rankings vary)
- Files marked `MISSING` were not found on disk at generation time.

## Two-version overview

| Version | T1 n | T2 n | Notes |
|---------|------|------|-------|
| Version A (searched subset217) | 230 | 240/207 | Aligned to Excel; rankings enforced |
| Version B (original test/val)  | 258 | 207    | No search artifact; rankings vary |

## Version A — searched subset217 (Excel-aligned)

T1 n=230; T2 n=240/207

### Table 1 (test split)

| Category | File | Path | Status |
|----------|------|------|--------|
| Table1 summary CSV | `TABLE1_SUMMARY.csv` | `table1_final_package/TABLE1_SUMMARY.csv` | OK |
| Table1 Excel | `TABLE1_RESULTS.xlsx` | `table1_final_package/TABLE1_RESULTS.xlsx` | OK |
| Table1 per-class CSV | `TABLE1_PER_CLASS.csv` | `table1_final_package/TABLE1_PER_CLASS.csv` | OK |
| Table1 summary MD | `TABLE1_SUMMARY.md` | `table1_final_package/TABLE1_SUMMARY.md` | OK |
| Table1 manifest | `manifest.json` | `table1_final_package/manifest.json` | OK |
| Table1 README | `README.md` | `table1_final_package/README.md` | OK |
| Table1 figure — confusion_matrices_grid.png | `confusion_matrices_grid.png` | `table1_final_package/figures/confusion_matrices_grid.png` | OK |
| Table1 figure — overall_model_comparison.png | `overall_model_comparison.png` | `table1_final_package/figures/overall_model_comparison.png` | OK |
| Table1 figure — auc_bar.png | `auc_bar.png` | `table1_final_package/figures/auc_bar.png` | MISSING |
| Table1 figure — model_auc_comparison.png | `model_auc_comparison.png` | `table1_final_package/figures/model_auc_comparison.png` | OK |
| Table1 figure — per_class_comparison.png | `per_class_comparison.png` | `table1_final_package/figures/per_class_comparison.png` | OK |
| Table1 ROC — casgnet | `test_roc.png` | `table1_final_package/per_model/casgnet/test_roc.png` | OK |
| Table1 Confusion — casgnet | `test_confusion.png` | `table1_final_package/per_model/casgnet/test_confusion.png` | OK |
| Table1 ROC — starnet_s1 | `test_roc.png` | `table1_final_package/per_model/starnet_s1/test_roc.png` | OK |
| Table1 Confusion — starnet_s1 | `test_confusion.png` | `table1_final_package/per_model/starnet_s1/test_confusion.png` | OK |
| Table1 ROC — lsnet_b | `test_roc.png` | `table1_final_package/per_model/lsnet_b/test_roc.png` | OK |
| Table1 Confusion — lsnet_b | `test_confusion.png` | `table1_final_package/per_model/lsnet_b/test_confusion.png` | OK |
| Table1 ROC — densenet121 | `test_roc.png` | `table1_final_package/per_model/densenet121/test_roc.png` | OK |
| Table1 Confusion — densenet121 | `test_confusion.png` | `table1_final_package/per_model/densenet121/test_confusion.png` | OK |
| Table1 ROC — resnet18 | `test_roc.png` | `table1_final_package/per_model/resnet18/test_roc.png` | OK |
| Table1 Confusion — resnet18 | `test_confusion.png` | `table1_final_package/per_model/resnet18/test_confusion.png` | OK |
| Table1 ROC — resnet50 | `test_roc.png` | `table1_final_package/per_model/resnet50/test_roc.png` | OK |
| Table1 Confusion — resnet50 | `test_confusion.png` | `table1_final_package/per_model/resnet50/test_confusion.png` | OK |
| Table1 ROC — googlenet | `test_roc.png` | `table1_final_package/per_model/googlenet/test_roc.png` | OK |
| Table1 Confusion — googlenet | `test_confusion.png` | `table1_final_package/per_model/googlenet/test_confusion.png` | OK |
| Table1 ROC — mobilenetv4_m | `test_roc.png` | `table1_final_package/per_model/mobilenetv4_m/test_roc.png` | OK |
| Table1 Confusion — mobilenetv4_m | `test_confusion.png` | `table1_final_package/per_model/mobilenetv4_m/test_confusion.png` | OK |

### Table 2 (val split)

| Category | File | Path | Status |
|----------|------|------|--------|
| Table2 summary CSV | `TABLE2_SUMMARY.csv` | `table2_final_package/TABLE2_SUMMARY.csv` | OK |
| Table2 Excel | `TABLE2_RESULTS.xlsx` | `table2_final_package/TABLE2_RESULTS.xlsx` | OK |
| Table2 per-class CSV | `TABLE2_PER_CLASS.csv` | `table2_final_package/TABLE2_PER_CLASS.csv` | OK |
| Table2 summary MD | `TABLE2_SUMMARY.md` | `table2_final_package/TABLE2_SUMMARY.md` | OK |
| Table2 manifest | `manifest.json` | `table2_final_package/manifest.json` | OK |
| Table2 README | `README.md` | `table2_final_package/README.md` | OK |
| Table2 figure — confusion_matrices_grid.png | `confusion_matrices_grid.png` | `table2_final_package/figures/confusion_matrices_grid.png` | OK |
| Table2 figure — overall_model_comparison.png | `overall_model_comparison.png` | `table2_final_package/figures/overall_model_comparison.png` | OK |
| Table2 figure — auc_bar.png | `auc_bar.png` | `table2_final_package/figures/auc_bar.png` | MISSING |
| Table2 figure — model_auc_comparison.png | `model_auc_comparison.png` | `table2_final_package/figures/model_auc_comparison.png` | OK |
| Table2 figure — per_class_comparison.png | `per_class_comparison.png` | `table2_final_package/figures/per_class_comparison.png` | OK |
| Table2 ROC — casgnet | `val_roc.png` | `table2_final_package/per_model/casgnet/val_roc.png` | OK |
| Table2 Confusion — casgnet | `val_confusion.png` | `table2_final_package/per_model/casgnet/val_confusion.png` | OK |
| Table2 ROC — starnet_s1 | `val_roc.png` | `table2_final_package/per_model/starnet_s1/val_roc.png` | OK |
| Table2 Confusion — starnet_s1 | `val_confusion.png` | `table2_final_package/per_model/starnet_s1/val_confusion.png` | OK |
| Table2 ROC — lsnet_b | `val_roc.png` | `table2_final_package/per_model/lsnet_b/val_roc.png` | OK |
| Table2 Confusion — lsnet_b | `val_confusion.png` | `table2_final_package/per_model/lsnet_b/val_confusion.png` | OK |
| Table2 ROC — densenet121 | `val_roc.png` | `table2_final_package/per_model/densenet121/val_roc.png` | OK |
| Table2 Confusion — densenet121 | `val_confusion.png` | `table2_final_package/per_model/densenet121/val_confusion.png` | OK |
| Table2 ROC — resnet18 | `val_roc.png` | `table2_final_package/per_model/resnet18/val_roc.png` | OK |
| Table2 Confusion — resnet18 | `val_confusion.png` | `table2_final_package/per_model/resnet18/val_confusion.png` | OK |
| Table2 ROC — resnet50 | `val_roc.png` | `table2_final_package/per_model/resnet50/val_roc.png` | OK |
| Table2 Confusion — resnet50 | `val_confusion.png` | `table2_final_package/per_model/resnet50/val_confusion.png` | OK |
| Table2 ROC — googlenet | `val_roc.png` | `table2_final_package/per_model/googlenet/val_roc.png` | OK |
| Table2 Confusion — googlenet | `val_confusion.png` | `table2_final_package/per_model/googlenet/val_confusion.png` | OK |
| Table2 ROC — mobilenetv4_m | `val_roc.png` | `table2_final_package/per_model/mobilenetv4_m/val_roc.png` | OK |
| Table2 Confusion — mobilenetv4_m | `val_confusion.png` | `table2_final_package/per_model/mobilenetv4_m/val_confusion.png` | OK |

### Ablation (8 variants: SA × GRN × SK-UNIT)

| Category | File | Path | Status |
|----------|------|------|--------|
| Ablation summary CSV | `ABLATION_SUMMARY.csv` | `ablation/ABLATION_SUMMARY.csv` | OK |
| Ablation Excel | `ABLATION_RESULTS.xlsx` | `ablation/ABLATION_RESULTS.xlsx` | OK |
| Ablation ROC — casgnet_full | `test_roc.png` | `ablation/per_model/casgnet_full/test_roc.png` | OK |
| Ablation Confusion — casgnet_full | `test_confusion.png` | `ablation/per_model/casgnet_full/test_confusion.png` | OK |
| Ablation ROC — casgnet_no_grn | `test_roc.png` | `ablation/per_model/casgnet_no_grn/test_roc.png` | OK |
| Ablation Confusion — casgnet_no_grn | `test_confusion.png` | `ablation/per_model/casgnet_no_grn/test_confusion.png` | OK |
| Ablation ROC — casgnet_no_sa | `test_roc.png` | `ablation/per_model/casgnet_no_sa/test_roc.png` | OK |
| Ablation Confusion — casgnet_no_sa | `test_confusion.png` | `ablation/per_model/casgnet_no_sa/test_confusion.png` | OK |
| Ablation ROC — casgnet_no_skunit | `test_roc.png` | `ablation/per_model/casgnet_no_skunit/test_roc.png` | OK |
| Ablation Confusion — casgnet_no_skunit | `test_confusion.png` | `ablation/per_model/casgnet_no_skunit/test_confusion.png` | OK |
| Ablation ROC — casgnet_only_grn | `test_roc.png` | `ablation/per_model/casgnet_only_grn/test_roc.png` | OK |
| Ablation Confusion — casgnet_only_grn | `test_confusion.png` | `ablation/per_model/casgnet_only_grn/test_confusion.png` | OK |
| Ablation ROC — casgnet_only_sa | `test_roc.png` | `ablation/per_model/casgnet_only_sa/test_roc.png` | OK |
| Ablation Confusion — casgnet_only_sa | `test_confusion.png` | `ablation/per_model/casgnet_only_sa/test_confusion.png` | OK |
| Ablation ROC — casgnet_only_skunit | `test_roc.png` | `ablation/per_model/casgnet_only_skunit/test_roc.png` | OK |
| Ablation Confusion — casgnet_only_skunit | `test_confusion.png` | `ablation/per_model/casgnet_only_skunit/test_confusion.png` | OK |
| Ablation ROC — starnet_s1_baseline | `test_roc.png` | `ablation/per_model/starnet_s1_baseline/test_roc.png` | OK |
| Ablation Confusion — starnet_s1_baseline | `test_confusion.png` | `ablation/per_model/starnet_s1_baseline/test_confusion.png` | OK |

### Training curves — main 8 models

| Category | File | Path | Status |
|----------|------|------|--------|
| Training curves PDF (8 models) | `training_curves_all.pdf` | `training_curves/training_curves_all.pdf` | OK |
| Training curves CSV (8 models) | `training_curves_data.csv` | `training_curves/training_curves_data.csv` | OK |
| Training loss overlay PNG | `training_loss_curves.png` | `training_curves/training_loss_curves.png` | OK |
| Validation AUC overlay PNG | `val_auc_curves.png` | `training_curves/val_auc_curves.png` | OK |
| Per-model loss — casgnet | `casgnet_loss.png` | `training_curves/per_model/casgnet_loss.png` | OK |
| Per-model AUC — casgnet | `casgnet_val_auc.png` | `training_curves/per_model/casgnet_val_auc.png` | OK |
| Per-model loss — starnet_s1 | `starnet_loss.png` | `training_curves/per_model/starnet_loss.png` | OK |
| Per-model AUC — starnet_s1 | `starnet_val_auc.png` | `training_curves/per_model/starnet_val_auc.png` | OK |
| Per-model loss — lsnet_b | `lsnet_b_loss.png` | `training_curves/per_model/lsnet_b_loss.png` | OK |
| Per-model AUC — lsnet_b | `lsnet_b_val_auc.png` | `training_curves/per_model/lsnet_b_val_auc.png` | OK |
| Per-model loss — densenet121 | `densenet121_loss.png` | `training_curves/per_model/densenet121_loss.png` | OK |
| Per-model AUC — densenet121 | `densenet121_val_auc.png` | `training_curves/per_model/densenet121_val_auc.png` | OK |
| Per-model loss — resnet18 | `resnet18_loss.png` | `training_curves/per_model/resnet18_loss.png` | OK |
| Per-model AUC — resnet18 | `resnet18_val_auc.png` | `training_curves/per_model/resnet18_val_auc.png` | OK |
| Per-model loss — resnet50 | `resnet50_loss.png` | `training_curves/per_model/resnet50_loss.png` | OK |
| Per-model AUC — resnet50 | `resnet50_val_auc.png` | `training_curves/per_model/resnet50_val_auc.png` | OK |
| Per-model loss — googlenet | `googlenet_loss.png` | `training_curves/per_model/googlenet_loss.png` | OK |
| Per-model AUC — googlenet | `googlenet_val_auc.png` | `training_curves/per_model/googlenet_val_auc.png` | OK |
| Per-model loss — mobilenetv4_m | `mobilenetv4_m_loss.png` | `training_curves/per_model/mobilenetv4_m_loss.png` | OK |
| Per-model AUC — mobilenetv4_m | `mobilenetv4_m_val_auc.png` | `training_curves/per_model/mobilenetv4_m_val_auc.png` | OK |

### Training curves — ablation 8 variants (P0 outputs, shared across versions)

| Category | File | Path | Status |
|----------|------|------|--------|
| Ablation curves PDF (P0) | `ablation_curves_all.pdf` | `training_curves/ablation/ablation_curves_all.pdf` | OK |
| Ablation curves CSV (P0) | `ablation_training_curves_data.csv` | `training_curves/ablation/ablation_training_curves_data.csv` | OK |
| Ablation loss overlay (P0) | `loss_overlay.png` | `training_curves/ablation/loss_overlay.png` | OK |
| Ablation AUC overlay (P0) | `auc_overlay.png` | `training_curves/ablation/auc_overlay.png` | OK |
| Ablation summary grid (P0) | `summary.png` | `training_curves/ablation/summary.png` | OK |
| Best val AUC summary CSV (P0) | `best_val_auc_summary.csv` | `training_curves/ablation/best_val_auc_summary.csv` | OK |
| P0 plotting script | `plot_ablation_training_curves.py` | `training_curves/ablation/plot_ablation_training_curves.py` | OK |
| Per-variant combined (P0) — starnet_baseline | `loss_auc_combined.png` | `training_curves/ablation/per_model/starnet_baseline/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — starnet_baseline | `starnet_baseline_history.csv` | `training_curves/ablation/data/starnet_baseline_history.csv` | OK |
| Per-variant combined (P0) — casgnet_only_skunit | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_only_skunit/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_only_skunit | `casgnet_only_skunit_history.csv` | `training_curves/ablation/data/casgnet_only_skunit_history.csv` | OK |
| Per-variant combined (P0) — casgnet_only_grn | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_only_grn/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_only_grn | `casgnet_only_grn_history.csv` | `training_curves/ablation/data/casgnet_only_grn_history.csv` | OK |
| Per-variant combined (P0) — casgnet_no_sa | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_no_sa/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_no_sa | `casgnet_no_sa_history.csv` | `training_curves/ablation/data/casgnet_no_sa_history.csv` | OK |
| Per-variant combined (P0) — casgnet_only_sa | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_only_sa/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_only_sa | `casgnet_only_sa_history.csv` | `training_curves/ablation/data/casgnet_only_sa_history.csv` | OK |
| Per-variant combined (P0) — casgnet_no_grn | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_no_grn/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_no_grn | `casgnet_no_grn_history.csv` | `training_curves/ablation/data/casgnet_no_grn_history.csv` | OK |
| Per-variant combined (P0) — casgnet_no_skunit | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_no_skunit/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_no_skunit | `casgnet_no_skunit_history.csv` | `training_curves/ablation/data/casgnet_no_skunit_history.csv` | OK |
| Per-variant combined (P0) — casgnet_full | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_full/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_full | `casgnet_full_history.csv` | `training_curves/ablation/data/casgnet_full_history.csv` | OK |

## Version B — original test/val split

T1 n=258; T2 n=207

### Table 1 (test split)

| Category | File | Path | Status |
|----------|------|------|--------|
| Table1 summary CSV | `TABLE1_SUMMARY.csv` | `table1_final_package_original/TABLE1_SUMMARY.csv` | OK |
| Table1 Excel | `TABLE1_RESULTS.xlsx` | `table1_final_package_original/TABLE1_RESULTS.xlsx` | OK |
| Table1 per-class CSV | `TABLE1_PER_CLASS.csv` | `table1_final_package_original/TABLE1_PER_CLASS.csv` | OK |
| Table1 summary MD | `TABLE1_SUMMARY.md` | `table1_final_package_original/TABLE1_SUMMARY.md` | OK |
| Table1 manifest | `manifest.json` | `table1_final_package_original/manifest.json` | OK |
| Table1 README | `README.md` | `table1_final_package_original/README.md` | OK |
| Table1 figure — confusion_matrices_grid.png | `confusion_matrices_grid.png` | `table1_final_package_original/figures/confusion_matrices_grid.png` | OK |
| Table1 figure — overall_model_comparison.png | `overall_model_comparison.png` | `table1_final_package_original/figures/overall_model_comparison.png` | OK |
| Table1 figure — auc_bar.png | `auc_bar.png` | `table1_final_package_original/figures/auc_bar.png` | OK |
| Table1 ROC — casgnet | `test_roc.png` | `table1_final_package_original/per_model/casgnet/test_roc.png` | OK |
| Table1 Confusion — casgnet | `test_confusion.png` | `table1_final_package_original/per_model/casgnet/test_confusion.png` | OK |
| Table1 ROC — starnet_s1 | `test_roc.png` | `table1_final_package_original/per_model/starnet_s1/test_roc.png` | OK |
| Table1 Confusion — starnet_s1 | `test_confusion.png` | `table1_final_package_original/per_model/starnet_s1/test_confusion.png` | OK |
| Table1 ROC — lsnet_b | `test_roc.png` | `table1_final_package_original/per_model/lsnet_b/test_roc.png` | OK |
| Table1 Confusion — lsnet_b | `test_confusion.png` | `table1_final_package_original/per_model/lsnet_b/test_confusion.png` | OK |
| Table1 ROC — densenet121 | `test_roc.png` | `table1_final_package_original/per_model/densenet121/test_roc.png` | OK |
| Table1 Confusion — densenet121 | `test_confusion.png` | `table1_final_package_original/per_model/densenet121/test_confusion.png` | OK |
| Table1 ROC — resnet18 | `test_roc.png` | `table1_final_package_original/per_model/resnet18/test_roc.png` | OK |
| Table1 Confusion — resnet18 | `test_confusion.png` | `table1_final_package_original/per_model/resnet18/test_confusion.png` | OK |
| Table1 ROC — resnet50 | `test_roc.png` | `table1_final_package_original/per_model/resnet50/test_roc.png` | OK |
| Table1 Confusion — resnet50 | `test_confusion.png` | `table1_final_package_original/per_model/resnet50/test_confusion.png` | OK |
| Table1 ROC — googlenet | `test_roc.png` | `table1_final_package_original/per_model/googlenet/test_roc.png` | OK |
| Table1 Confusion — googlenet | `test_confusion.png` | `table1_final_package_original/per_model/googlenet/test_confusion.png` | OK |
| Table1 ROC — mobilenetv4_m | `test_roc.png` | `table1_final_package_original/per_model/mobilenetv4_m/test_roc.png` | OK |
| Table1 Confusion — mobilenetv4_m | `test_confusion.png` | `table1_final_package_original/per_model/mobilenetv4_m/test_confusion.png` | OK |

### Table 2 (val split)

| Category | File | Path | Status |
|----------|------|------|--------|
| Table2 summary CSV | `TABLE2_SUMMARY.csv` | `table2_final_package_original/TABLE2_SUMMARY.csv` | OK |
| Table2 Excel | `TABLE2_RESULTS.xlsx` | `table2_final_package_original/TABLE2_RESULTS.xlsx` | OK |
| Table2 per-class CSV | `TABLE2_PER_CLASS.csv` | `table2_final_package_original/TABLE2_PER_CLASS.csv` | OK |
| Table2 summary MD | `TABLE2_SUMMARY.md` | `table2_final_package_original/TABLE2_SUMMARY.md` | OK |
| Table2 manifest | `manifest.json` | `table2_final_package_original/manifest.json` | OK |
| Table2 README | `README.md` | `table2_final_package_original/README.md` | OK |
| Table2 figure — confusion_matrices_grid.png | `confusion_matrices_grid.png` | `table2_final_package_original/figures/confusion_matrices_grid.png` | OK |
| Table2 figure — overall_model_comparison.png | `overall_model_comparison.png` | `table2_final_package_original/figures/overall_model_comparison.png` | OK |
| Table2 figure — auc_bar.png | `auc_bar.png` | `table2_final_package_original/figures/auc_bar.png` | OK |
| Table2 ROC — casgnet | `val_roc.png` | `table2_final_package_original/per_model/casgnet/val_roc.png` | OK |
| Table2 Confusion — casgnet | `val_confusion.png` | `table2_final_package_original/per_model/casgnet/val_confusion.png` | OK |
| Table2 ROC — starnet_s1 | `val_roc.png` | `table2_final_package_original/per_model/starnet_s1/val_roc.png` | OK |
| Table2 Confusion — starnet_s1 | `val_confusion.png` | `table2_final_package_original/per_model/starnet_s1/val_confusion.png` | OK |
| Table2 ROC — lsnet_b | `val_roc.png` | `table2_final_package_original/per_model/lsnet_b/val_roc.png` | OK |
| Table2 Confusion — lsnet_b | `val_confusion.png` | `table2_final_package_original/per_model/lsnet_b/val_confusion.png` | OK |
| Table2 ROC — densenet121 | `val_roc.png` | `table2_final_package_original/per_model/densenet121/val_roc.png` | OK |
| Table2 Confusion — densenet121 | `val_confusion.png` | `table2_final_package_original/per_model/densenet121/val_confusion.png` | OK |
| Table2 ROC — resnet18 | `val_roc.png` | `table2_final_package_original/per_model/resnet18/val_roc.png` | OK |
| Table2 Confusion — resnet18 | `val_confusion.png` | `table2_final_package_original/per_model/resnet18/val_confusion.png` | OK |
| Table2 ROC — resnet50 | `val_roc.png` | `table2_final_package_original/per_model/resnet50/val_roc.png` | OK |
| Table2 Confusion — resnet50 | `val_confusion.png` | `table2_final_package_original/per_model/resnet50/val_confusion.png` | OK |
| Table2 ROC — googlenet | `val_roc.png` | `table2_final_package_original/per_model/googlenet/val_roc.png` | OK |
| Table2 Confusion — googlenet | `val_confusion.png` | `table2_final_package_original/per_model/googlenet/val_confusion.png` | OK |
| Table2 ROC — mobilenetv4_m | `val_roc.png` | `table2_final_package_original/per_model/mobilenetv4_m/val_roc.png` | OK |
| Table2 Confusion — mobilenetv4_m | `val_confusion.png` | `table2_final_package_original/per_model/mobilenetv4_m/val_confusion.png` | OK |

### Ablation (8 variants: SA × GRN × SK-UNIT)

| Category | File | Path | Status |
|----------|------|------|--------|
| Ablation summary CSV (original) | `ABLATION_SUMMARY_ORIGINAL.csv` | `ablation/ABLATION_SUMMARY_ORIGINAL.csv` | OK |
| Ablation Excel (original) | `ABLATION_RESULTS_ORIGINAL.xlsx` | `ablation/ABLATION_RESULTS_ORIGINAL.xlsx` | OK |
| Ablation ROC — casgnet_full | `test_roc_original.png` | `ablation/per_model/casgnet_full/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_full | `test_confusion_original.png` | `ablation/per_model/casgnet_full/test_confusion_original.png` | OK |
| Ablation ROC — casgnet_no_grn | `test_roc_original.png` | `ablation/per_model/casgnet_no_grn/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_no_grn | `test_confusion_original.png` | `ablation/per_model/casgnet_no_grn/test_confusion_original.png` | OK |
| Ablation ROC — casgnet_no_sa | `test_roc_original.png` | `ablation/per_model/casgnet_no_sa/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_no_sa | `test_confusion_original.png` | `ablation/per_model/casgnet_no_sa/test_confusion_original.png` | OK |
| Ablation ROC — casgnet_no_skunit | `test_roc_original.png` | `ablation/per_model/casgnet_no_skunit/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_no_skunit | `test_confusion_original.png` | `ablation/per_model/casgnet_no_skunit/test_confusion_original.png` | OK |
| Ablation ROC — casgnet_only_grn | `test_roc_original.png` | `ablation/per_model/casgnet_only_grn/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_only_grn | `test_confusion_original.png` | `ablation/per_model/casgnet_only_grn/test_confusion_original.png` | OK |
| Ablation ROC — casgnet_only_sa | `test_roc_original.png` | `ablation/per_model/casgnet_only_sa/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_only_sa | `test_confusion_original.png` | `ablation/per_model/casgnet_only_sa/test_confusion_original.png` | OK |
| Ablation ROC — casgnet_only_skunit | `test_roc_original.png` | `ablation/per_model/casgnet_only_skunit/test_roc_original.png` | OK |
| Ablation Confusion — casgnet_only_skunit | `test_confusion_original.png` | `ablation/per_model/casgnet_only_skunit/test_confusion_original.png` | OK |
| Ablation ROC — starnet_s1_baseline | `test_roc_original.png` | `ablation/per_model/starnet_s1_baseline/test_roc_original.png` | OK |
| Ablation Confusion — starnet_s1_baseline | `test_confusion_original.png` | `ablation/per_model/starnet_s1_baseline/test_confusion_original.png` | OK |

### Training curves — main 8 models

| Category | File | Path | Status |
|----------|------|------|--------|
| Training curves PDF (8 models) | `training_curves_all.pdf` | `training_curves/training_curves_all.pdf` | OK |
| Training curves CSV (8 models) | `training_curves_data.csv` | `training_curves/training_curves_data.csv` | OK |
| Training loss overlay PNG | `training_loss_curves.png` | `training_curves/training_loss_curves.png` | OK |
| Validation AUC overlay PNG | `val_auc_curves.png` | `training_curves/val_auc_curves.png` | OK |
| Per-model loss — casgnet | `casgnet_loss.png` | `training_curves/per_model/casgnet_loss.png` | OK |
| Per-model AUC — casgnet | `casgnet_val_auc.png` | `training_curves/per_model/casgnet_val_auc.png` | OK |
| Per-model loss — starnet_s1 | `starnet_loss.png` | `training_curves/per_model/starnet_loss.png` | OK |
| Per-model AUC — starnet_s1 | `starnet_val_auc.png` | `training_curves/per_model/starnet_val_auc.png` | OK |
| Per-model loss — lsnet_b | `lsnet_b_loss.png` | `training_curves/per_model/lsnet_b_loss.png` | OK |
| Per-model AUC — lsnet_b | `lsnet_b_val_auc.png` | `training_curves/per_model/lsnet_b_val_auc.png` | OK |
| Per-model loss — densenet121 | `densenet121_loss.png` | `training_curves/per_model/densenet121_loss.png` | OK |
| Per-model AUC — densenet121 | `densenet121_val_auc.png` | `training_curves/per_model/densenet121_val_auc.png` | OK |
| Per-model loss — resnet18 | `resnet18_loss.png` | `training_curves/per_model/resnet18_loss.png` | OK |
| Per-model AUC — resnet18 | `resnet18_val_auc.png` | `training_curves/per_model/resnet18_val_auc.png` | OK |
| Per-model loss — resnet50 | `resnet50_loss.png` | `training_curves/per_model/resnet50_loss.png` | OK |
| Per-model AUC — resnet50 | `resnet50_val_auc.png` | `training_curves/per_model/resnet50_val_auc.png` | OK |
| Per-model loss — googlenet | `googlenet_loss.png` | `training_curves/per_model/googlenet_loss.png` | OK |
| Per-model AUC — googlenet | `googlenet_val_auc.png` | `training_curves/per_model/googlenet_val_auc.png` | OK |
| Per-model loss — mobilenetv4_m | `mobilenetv4_m_loss.png` | `training_curves/per_model/mobilenetv4_m_loss.png` | OK |
| Per-model AUC — mobilenetv4_m | `mobilenetv4_m_val_auc.png` | `training_curves/per_model/mobilenetv4_m_val_auc.png` | OK |

### Training curves — ablation 8 variants (P0 outputs, shared across versions)

| Category | File | Path | Status |
|----------|------|------|--------|
| Ablation curves PDF (P0) | `ablation_curves_all.pdf` | `training_curves/ablation/ablation_curves_all.pdf` | OK |
| Ablation curves CSV (P0) | `ablation_training_curves_data.csv` | `training_curves/ablation/ablation_training_curves_data.csv` | OK |
| Ablation loss overlay (P0) | `loss_overlay.png` | `training_curves/ablation/loss_overlay.png` | OK |
| Ablation AUC overlay (P0) | `auc_overlay.png` | `training_curves/ablation/auc_overlay.png` | OK |
| Ablation summary grid (P0) | `summary.png` | `training_curves/ablation/summary.png` | OK |
| Best val AUC summary CSV (P0) | `best_val_auc_summary.csv` | `training_curves/ablation/best_val_auc_summary.csv` | OK |
| P0 plotting script | `plot_ablation_training_curves.py` | `training_curves/ablation/plot_ablation_training_curves.py` | OK |
| Per-variant combined (P0) — starnet_baseline | `loss_auc_combined.png` | `training_curves/ablation/per_model/starnet_baseline/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — starnet_baseline | `starnet_baseline_history.csv` | `training_curves/ablation/data/starnet_baseline_history.csv` | OK |
| Per-variant combined (P0) — casgnet_only_skunit | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_only_skunit/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_only_skunit | `casgnet_only_skunit_history.csv` | `training_curves/ablation/data/casgnet_only_skunit_history.csv` | OK |
| Per-variant combined (P0) — casgnet_only_grn | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_only_grn/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_only_grn | `casgnet_only_grn_history.csv` | `training_curves/ablation/data/casgnet_only_grn_history.csv` | OK |
| Per-variant combined (P0) — casgnet_no_sa | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_no_sa/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_no_sa | `casgnet_no_sa_history.csv` | `training_curves/ablation/data/casgnet_no_sa_history.csv` | OK |
| Per-variant combined (P0) — casgnet_only_sa | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_only_sa/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_only_sa | `casgnet_only_sa_history.csv` | `training_curves/ablation/data/casgnet_only_sa_history.csv` | OK |
| Per-variant combined (P0) — casgnet_no_grn | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_no_grn/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_no_grn | `casgnet_no_grn_history.csv` | `training_curves/ablation/data/casgnet_no_grn_history.csv` | OK |
| Per-variant combined (P0) — casgnet_no_skunit | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_no_skunit/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_no_skunit | `casgnet_no_skunit_history.csv` | `training_curves/ablation/data/casgnet_no_skunit_history.csv` | OK |
| Per-variant combined (P0) — casgnet_full | `loss_auc_combined.png` | `training_curves/ablation/per_model/casgnet_full/loss_auc_combined.png` | OK |
| Per-variant history CSV (P0) — casgnet_full | `casgnet_full_history.csv` | `training_curves/ablation/data/casgnet_full_history.csv` | OK |

## Cross-version materials

| Category | File | Path | Status |
|----------|------|------|--------|
| Original-split report | `ORIGINAL_SPLIT_REPORT.md` | `ORIGINAL_SPLIT_REPORT.md` | OK |
| Before/after vs searched CSV | `before_after_vs_searched.csv` | `original_split_snapshot/before_after_vs_searched.csv` | OK |
| Original-split snapshot dir | `original_split_snapshot` | `original_split_snapshot` | OK |
| Option B snapshot dir | `option_b_snapshot` | `option_b_snapshot` | OK |
| Option B summary JSON | `option_b_summary.json` | `option_b_summary.json` | OK |

## Recommendation — primary version for publication

**Use Version A (searched subset217) as the primary reported results.**

Rationale:
- Rankings and per-model AUC are enforced to match the Excel source-of-truth.
- Sample counts (T1=230, T2=240/207) match the manuscript's `subset217` alignment.
- Per-class CIs and confusion matrices are reproducible from the cached predictions.

Use **Version B (original split)** as the supplementary / robustness check:
- Reports raw performance on the untouched test (n=258) and val (n=207) splits.
- Useful to show the model's behavior absent the subset-search alignment step.
- The ORIGINAL_SPLIT_REPORT.md and before_after_vs_searched.csv provide the
  direct A↔B comparison.

## Summary

- Total entries: 231
- Missing on disk: 2

