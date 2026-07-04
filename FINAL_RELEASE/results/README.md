# PUBLICATION_PACKAGE

This package aggregates all publication materials for the **hip implant X-ray
classification** paper. Every file in this package is a **real file copy**
(`cp` / `cp -r`) of the already-complete artefacts living under
`evaluation_results/excel_aligned/` — no plots, data, or binaries have been
regenerated or modified, and no symlinks are used. The only file written fresh
is this `README.md`.

## Recommendation (per reviewer assessment)

**Use Version B (original test/val split) as the primary results in the
manuscript.** Version A (subset217 searched) is retained as supplementary /
reference material only.

| Priority | Version | Directory | Role |
|---|---|---|---|
| **Primary** | Version B — original | `version_B_original/` | Main Tables 1 & 2, ablation, figures, DeLong tests, cross-version report |
| Supplementary | Version A — searched | `version_A_searched/` | Excel-aligned subset217 results; reference only — not for main claims |

See `REVIEWER_ASSESSMENT.md` for the full rationale (test-set optimization
risk in Version A; CasGNet #1 still holds in Version B on both T1 and T2).

## Two data versions

| Version | Directory | Description |
|---|---|---|
| **Version B — original** *(primary)* | `version_B_original/` | Results on the **original** test/val split (258 test / 207 val). No subset search. Recommended for publication. |
| **Version A — searched** *(supplementary)* | `version_A_searched/` | Results on the **subset217 searched** test/val split (Excel-aligned, rankings enforced). Reference / transparency only. |

Training curves (main 8-model comparison) are **shared** between the two
versions (training data does not depend on the test/val split). Ablation
training curves are likewise duplicated in both versions.

## Top-level materials

| File | Description |
|---|---|
| `README.md` | This file |
| `PUBLICATION_MATERIALS_INDEX.md` | Master index of all publication artefacts (source paths, status) |
| `REVIEWER_ASSESSMENT.md` | Peer-review-style assessment comparing Version A vs B; recommends Version B |

## Folder tree

```
PUBLICATION_PACKAGE/
├── README.md
├── PUBLICATION_MATERIALS_INDEX.md
├── REVIEWER_ASSESSMENT.md
│
├── version_B_original/                    ★ PRIMARY
│   ├── table1/
│   │   ├── TABLE1_SUMMARY.csv / .md / TABLE1_PER_CLASS.csv / TABLE1_RESULTS.xlsx
│   │   ├── per_model/                     8 models × ROC + confusion + metrics
│   │   ├── figures/                       confusion_matrices_grid, auc_bar,
│   │   │                                  overall_model_comparison (+ README)
│   │   ├── plots/                         per-model test_auc + confusion PNGs/CSVs
│   │   └── caches/
│   ├── table2/
│   │   ├── TABLE2_SUMMARY.csv / .md / TABLE2_PER_CLASS.csv / TABLE2_RESULTS.xlsx
│   │   ├── per_model/
│   │   ├── figures/                       confusion_matrices_grid, auc_bar,
│   │   │                                  overall_model_comparison (+ README)
│   │   ├── plots/
│   │   └── caches/
│   ├── ablation/
│   │   ├── ABLATION_SUMMARY_ORIGINAL.csv / ABLATION_RESULTS_ORIGINAL.xlsx
│   │   └── per_model/                     8 variants × original-split metrics + plots
│   ├── training_curves/
│   │   ├── training_curves_all.pdf        8-model main comparison
│   │   ├── training_curves_data.csv
│   │   ├── data/ per_model/ per_model_combined/ casgnet_vs_starnet/ annotated/
│   │   └── ablation/                      ★ ablation training curves (8 variants)
│   │       ├── ablation_curves_all.pdf
│   │       ├── ablation_training_curves_data.csv
│   │       ├── auc_overlay.png / loss_overlay.png / summary.png
│   │       ├── best_val_auc_summary.csv
│   │       ├── data/ per_model/
│   │       └── plot_ablation_training_curves.py
│   ├── delong_tests/                      ★ DeLong statistical tests (Version B only)
│   │   ├── DELONG_RESULTS.xlsx
│   │   ├── DELONG_TABLE1_TEST.csv / DELONG_TABLE2_VAL.csv
│   │   ├── DELONG_PER_CLASS_TABLE1.csv / DELONG_PER_CLASS_TABLE2.csv
│   │   ├── run_delong_tests.py
│   │   └── verification.json
│   └── cross_version/
│       ├── ORIGINAL_SPLIT_REPORT.md
│       └── before_after_vs_searched.csv
│
└── version_A_searched/                    supplementary / reference
    ├── table1/  table2/  ablation/         (searched subset217 results)
    └── training_curves/
        ├── … (same 8-model curves as Version B)
        └── ablation/                      ★ ablation training curves (duplicate)
```

## Per-version material index

### Version B — original *(primary)* (`version_B_original/`)

**Table 1 (original split, test-set metrics, n=258)**
- Summary / per-class CSV: `table1/TABLE1_SUMMARY.csv`, `table1/TABLE1_PER_CLASS.csv`
- Excel: `table1/TABLE1_RESULTS.xlsx`
- Per-model ROC + confusion: `table1/per_model/{model}/test_roc.png`,
  `table1/per_model/{model}/test_confusion.png`
- **Figures**: `table1/figures/` — `confusion_matrices_grid.png`, `auc_bar.png`,
  `overall_model_comparison.png`
- Additional plots: `table1/plots/`

**Table 2 (original split, val-set metrics, n=207)**
- Summary / per-class CSV: `table2/TABLE2_SUMMARY.csv`, `table2/TABLE2_PER_CLASS.csv`
- Excel: `table2/TABLE2_RESULTS.xlsx`
- Per-model ROC + confusion: `table2/per_model/{model}/val_roc.png`,
  `table2/per_model/{model}/val_confusion.png`
- **Figures**: `table2/figures/` — `confusion_matrices_grid.png`, `auc_bar.png`,
  `overall_model_comparison.png`
- Additional plots: `table2/plots/`

**Ablation (original split, n=258)**
- Summary CSV: `ablation/ABLATION_SUMMARY_ORIGINAL.csv`
- Excel: `ablation/ABLATION_RESULTS_ORIGINAL.xlsx`
- Per-variant: `ablation/per_model/{variant}/test_roc_original.png`,
  `test_confusion_original.png`, `metrics_per_class_original.csv`, `metrics.json`

**Training curves — main (8 models)**
- Combined PDF: `training_curves/training_curves_all.pdf`
- Combined CSV: `training_curves/training_curves_data.csv`
- Per-model history: `training_curves/data/{model}_history.csv`
- 2-model comparison: `training_curves/casgnet_vs_starnet/`

**Training curves — ablation (8 variants)**
- Combined PDF: `training_curves/ablation/ablation_curves_all.pdf`
- Data CSV: `training_curves/ablation/ablation_training_curves_data.csv`
- Overlays: `training_curves/ablation/auc_overlay.png`, `loss_overlay.png`, `summary.png`
- Per-variant history: `training_curves/ablation/data/{variant}_history.csv`

**DeLong tests** (`delong_tests/`)
- Excel summary: `DELONG_RESULTS.xlsx`
- Overall tests: `DELONG_TABLE1_TEST.csv`, `DELONG_TABLE2_VAL.csv`
- Per-class tests: `DELONG_PER_CLASS_TABLE1.csv`, `DELONG_PER_CLASS_TABLE2.csv`
- Repro script: `run_delong_tests.py`

**Cross-version comparison** (`cross_version/`)
- `ORIGINAL_SPLIT_REPORT.md` — narrative report vs searched split
- `before_after_vs_searched.csv` — per-model AUC before/after comparison

### Version A — searched *(supplementary)* (`version_A_searched/`)

**Table 1 (subset217 searched, test-set metrics, n=230)**
- Summary / Excel / per-model plots as in Version B structure
- Figures: `table1/figures/` (confusion grid, model AUC comparison,
  overall comparison, per-class comparison)

**Table 2 (subset217 searched, val-set metrics)**
- Same structure; figures in `table2/figures/`

**Ablation (subset217 searched, n=230)**
- Summary: `ablation/ABLATION_SUMMARY.csv`, `ablation/ABLATION_RESULTS.xlsx`

**Training curves**
- Main 8-model curves (duplicate of Version B)
- Ablation curves: `training_curves/ablation/` (duplicate of Version B)

## Publication checklist

### Ready in this package

- [x] Version B Table 1 & 2 summaries, Excel, per-model ROC/confusion
- [x] Version B Table 1 & 2 aggregate figures (`table1/figures/`, `table2/figures/`)
- [x] Version B ablation tables and per-variant plots
- [x] Main 8-model training curves (`training_curves_all.pdf`)
- [x] Ablation training curves (`training_curves/ablation/ablation_curves_all.pdf`)
- [x] DeLong test results (`delong_tests/DELONG_RESULTS.xlsx`)
- [x] Cross-version robustness report (`cross_version/`)
- [x] Version A supplementary results (full copy)
- [x] Master materials index (`PUBLICATION_MATERIALS_INDEX.md`)
- [x] Reviewer assessment with Version B recommendation (`REVIEWER_ASSESSMENT.md`)

### Author still needs to do (manuscript writing)

- [ ] **Interpret DeLong results** — report p-values in manuscript; if CasGNet vs
  #2 is not significant (p>0.05), revise wording to "highest mean AUC, not
  statistically distinguishable from densenet121 / lsnet_b / starnet_s1"
- [ ] **Update ranking statement** — replace "CasGNet #1, StarNet #2, lsnet_b #3"
  with "CasGNet ranks #1 on both test and val; #2/#3 positions vary and are not
  statistically stable"
- [ ] **Reframe ablation narrative** — SK-UNIT as primary contribution; SA/GRN
  require co-activation (only_sa inversion in Version B)
- [ ] **Add Limitations** — small sample per class, wide bootstrap CIs on minority
  classes, SENS/PPV trade-off (CasGNet lower SENS than densenet)
- [ ] **Cite Version A as supplementary** — label subset217 as Excel-alignment
  reference, not primary evidence

## Package statistics

- Total real files: **602**
- Total directories: **109**
- Total package size: **~47 MB**
- Symlinks used: **0** (all real file copies)

## MISSING items

None. Every source path in the integration mapping existed and was successfully
copied as real files.
