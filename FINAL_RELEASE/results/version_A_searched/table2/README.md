# Table 2 Final Package

**表二 独立测试集** — val_207 group (n=207, per-model subsets from train+val pool), v3 checkpoints, legacy_val_resize.

## Contents

| File | Description |
|------|-------------|
| `TABLE2_SUMMARY.csv` / `.md` | 8 models overall ACC/AUC vs Excel |
| `TABLE2_PER_CLASS.csv` | All models × all classes |
| `TABLE2_RESULTS.xlsx` | Excel export: 总体指标, 每类指标, 排名 (run `export_table2_excel.py`) |
| `per_model/<model>/` | Per-model metrics + `val_roc.png` + `val_confusion.png` |
| `manifest.json` | Package completeness manifest |

Excel source: sheet 3–4 of `整体实验结果_优化排版.xlsx` (see `../EXCEL_SHEET_MAPPING.md`).

## Settings

- Bootstrap: n=1000, seed=42
- Preprocessing: Resize 224×224 (`legacy_val_resize`)
- Checkpoints: `checkpoints/old_data_supcon_compare_v3/*/best_auc_model.pth`

## Models (8/8 complete)

- ✓ `casgnet`
- ✓ `mobilenetv4_m`
- ✓ `starnet_s1`
- ✓ `densenet121`
- ✓ `resnet18`
- ✓ `googlenet`
- ✓ `resnet50`
- ✓ `lsnet_b`
