# TABLE2 Final Package (ORIGINAL split)

**TABLE2 val set** — ORIGINAL `old_data/val` (n=207), v3 checkpoints, legacy_val_resize.
**No subset search.** All images in the split are used.

## Contents

| File | Description |
|------|-------------|
| `TABLE2_SUMMARY.csv` / `.md` | 8 models, 6 metrics + 95% CI |
| `TABLE2_PER_CLASS.csv` | 56 rows (8 models x 7 classes) |
| `TABLE2_RESULTS.xlsx` | Excel: 总体指标 / 每类指标 / 排名 |
| `per_model/<model>/` | metrics + `val_roc.png` + `val_confusion.png` |
| `caches/<model>_val_predictions.npz` | Filtered prediction cache |
| `manifest.json` | Package manifest |

## Settings

- Bootstrap: n=1000, seed=42, 95% CI
- Preprocessing: Resize 224x224 (legacy_val_resize)
- Checkpoints: `checkpoints/old_data_supcon_compare_v3/*/best_auc_model.pth`

## AUC ranking

1. casgnet — AUC 0.944(0.925-0.962)
2. mobilenetv4_m — AUC 0.936(0.914-0.956)
3. starnet_s1 — AUC 0.935(0.911-0.956)
4. resnet18 — AUC 0.934(0.913-0.953)
5. densenet121 — AUC 0.933(0.903-0.957)
6. googlenet — AUC 0.929(0.905-0.952)
7. resnet50 — AUC 0.922(0.892-0.948)
8. lsnet_b — AUC 0.922(0.898-0.945)
