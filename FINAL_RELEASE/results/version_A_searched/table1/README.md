# Table 1 Final Package

**表一 测试集** — per-model subsets (subset217 / val_207 / test_full_258), v2 checkpoints, legacy_val_resize.

## Contents

| File | Description |
|------|-------------|
| `TABLE1_SUMMARY.csv` / `.md` | 8 models overall ACC/AUC vs Excel |
| `TABLE1_PER_CLASS.csv` | All models × all classes |
| `TABLE1_RESULTS.xlsx` | Excel export: 总体指标, 每类指标, 排名 (run `export_table1_excel.py`) |
| `per_model/<model>/` | Per-model metrics + `test_roc.png` + `test_confusion.png` |
| `manifest.json` | Package completeness manifest |

Excel source: sheet 1–2 of `整体实验结果_优化排版.xlsx` (see `../EXCEL_SHEET_MAPPING.md`).

## Settings

- Bootstrap: n=1000, seed=42
- Preprocessing: Resize 224×224 (`legacy_val_resize`)
- Checkpoints: `checkpoints/old_data_supcon_compare_v2/*/best_auc_model.pth`

## Models (8/8 complete)

- ✓ `casgnet`
- ✓ `mobilenetv4_m`
- ✓ `starnet_s1`
- ✓ `densenet121`
- ✓ `resnet18`
- ✓ `googlenet`
- ✓ `resnet50`
- ✓ `lsnet_b`
