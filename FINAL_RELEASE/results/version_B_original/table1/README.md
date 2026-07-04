# TABLE1 Final Package (ORIGINAL split)

**TABLE1 test set** — ORIGINAL `old_data/test` (n=258), v2 checkpoints, legacy_val_resize.
**No subset search.** All images in the split are used.

## Contents

| File | Description |
|------|-------------|
| `TABLE1_SUMMARY.csv` / `.md` | 8 models, 6 metrics + 95% CI |
| `TABLE1_PER_CLASS.csv` | 56 rows (8 models x 7 classes) |
| `TABLE1_RESULTS.xlsx` | Excel: 总体指标 / 每类指标 / 排名 |
| `per_model/<model>/` | metrics + `test_roc.png` + `test_confusion.png` |
| `caches/<model>_test_predictions.npz` | Filtered prediction cache |
| `manifest.json` | Package manifest |

## Settings

- Bootstrap: n=1000, seed=42, 95% CI
- Preprocessing: Resize 224x224 (legacy_val_resize)
- Checkpoints: `checkpoints/old_data_supcon_compare_v2/*/best_auc_model.pth`

## AUC ranking

1. casgnet — AUC 0.962(0.948-0.976)
2. densenet121 — AUC 0.957(0.936-0.975)
3. lsnet_b — AUC 0.953(0.931-0.970)
4. starnet_s1 — AUC 0.952(0.929-0.972)
5. resnet18 — AUC 0.951(0.934-0.968)
6. mobilenetv4_m — AUC 0.918(0.893-0.940)
7. resnet50 — AUC 0.917(0.885-0.945)
8. googlenet — AUC 0.903(0.875-0.928)
