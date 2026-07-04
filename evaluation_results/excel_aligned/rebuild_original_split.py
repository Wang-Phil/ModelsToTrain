#!/usr/bin/env python3
"""
Option B (final): abandon subset217 search; use ORIGINAL old_data/test (n=258)
and old_data/val (n=207) splits for Table1 / Table2.

Predictions are extracted from the existing pool caches (no re-inference):
  - Table1: table1_per_model/caches/{model}_test_pool_predictions.npz (v2 ckpts)
            filter split_tags == "test" -> n=258
  - Table2: caches/{model}_val_pool_predictions.npz             (v3 ckpts)
            filter by path containing "/val/"   -> n=207

Outputs (NEW locations; existing searched packages are left untouched):
  table1_final_package_original/
  table2_final_package_original/
  original_split_snapshot/

Usage (project root):
  python evaluation_results/excel_aligned/rebuild_original_split.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from compare_models_on_eltra_test import row_eltra_bootstrap  # noqa: E402
from refresh_supcon_checkpoint_metrics import (  # noqa: E402
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
)
from run_all_models_eval import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXCEL_MODELS,
    N_BOOTSTRAP,
    bootstrap_per_class_auc_rows,
    generate_plots_for_cache,
    load_cache,
    macro_metrics_row,
    parse_point,
    plot_confusion,
    plot_roc,
    save_cache,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
MODELS = [m for m, _ in EXCEL_MODELS]
CK_MAP = {m: ck for m, ck in EXCEL_MODELS}

T1_POOL_DIR = HERE / "table1_per_model" / "caches"
T2_POOL_DIR = HERE / "caches"

T1_OUT = HERE / "table1_final_package_original"
T2_OUT = HERE / "table2_final_package_original"
SNAP_DIR = HERE / "original_split_snapshot"

OLD_DATA = ROOT / "old_data"
TEST_DIR = OLD_DATA / "test"
VAL_DIR = OLD_DATA / "val"


# ----------------------------------------------------------------------------
# Split identification
# ----------------------------------------------------------------------------
def list_split_class_counts(split_dir: Path) -> tuple[int, dict[str, int]]:
    counts: dict[str, int] = {}
    total = 0
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        n = sum(1 for _ in cls_dir.rglob("*") if _.is_file())
        counts[cls_dir.name] = n
        total += n
    return total, counts


def collect_split_image_list(split_dir: Path) -> list[str]:
    paths: list[str] = []
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.rglob("*")):
            if f.is_file():
                paths.append(str(f.resolve().as_posix()))
    return paths


# ----------------------------------------------------------------------------
# Pool cache loading + filtering
# ----------------------------------------------------------------------------
def load_pool(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    return {
        "probs": d["probs"],
        "yt": d["yt"],
        "yhat": d["yhat"],
        "class_names": [str(x) for x in d["class_names"].tolist()],
        "paths": d["paths"].astype(str) if "paths" in d.files else None,
        "split_tags": d["split_tags"].astype(str) if "split_tags" in d.files else None,
    }


def filter_test(pool: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Filter to test split (n=258) using split_tags (preferred) or path match."""
    if pool["split_tags"] is not None:
        mask = pool["split_tags"] == "test"
    else:
        assert pool["paths"] is not None
        mask = np.array(["/test/" in p for p in pool["paths"]])
    idx = np.where(mask)[0]
    return pool["probs"][idx], pool["yt"][idx], pool["yhat"][idx], pool["class_names"], idx


def filter_val(pool: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    if pool["split_tags"] is not None:
        mask = pool["split_tags"] == "val"
    else:
        assert pool["paths"] is not None
        mask = np.array(["/val/" in p for p in pool["paths"]])
    idx = np.where(mask)[0]
    return pool["probs"][idx], pool["yt"][idx], pool["yhat"][idx], pool["class_names"], idx


# ----------------------------------------------------------------------------
# Metric computation
# ----------------------------------------------------------------------------
def fmt_ci(v: float, lo: float, hi: float) -> str:
    return f"{v:.3f}({lo:.3f}-{hi:.3f})"


def compute_metrics_block(
    excel_name: str,
    ck_name: str,
    split: str,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> tuple[dict, list[dict]]:
    """Macro row (with bootstrap CI) + per-class rows."""
    n_cls = len(class_names)
    row = row_eltra_bootstrap(
        ck_name, yt, yhat, probs, n_cls,
        n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED,
    )
    row.pop("_detail", None)
    row.pop("_auc_sort", None)
    row["excel_model"] = excel_name
    row["split"] = split
    row["n_samples"] = int(len(yt))

    _, per_class_pt = None, None
    from train_casgnet_contrastive_newdata import compute_macro_classification_metrics
    _, per_class_pt = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    aucs_ovr_pt = _per_class_auc_ovr(yt, probs, n_cls)
    pc_rows = bootstrap_per_class_comparison_rows(
        yt, yhat, probs, per_class_pt, aucs_ovr_pt, class_names,
        n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED,
    )
    for pr in pc_rows:
        pr["excel_model"] = excel_name
        pr["split"] = split
    return row, pc_rows


# ----------------------------------------------------------------------------
# Plot generation
# ----------------------------------------------------------------------------
def make_plots(
    out_dir: Path,
    excel_name: str,
    split: str,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_rows = bootstrap_per_class_auc_rows(
        yt, probs, class_names, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )
    auc_csv = out_dir / f"{excel_name}_{split}_auc.csv"
    with auc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["CLASS", "AUC"])
        w.writeheader()
        for r in class_rows:
            w.writerow({"CLASS": r["CLASS"], "AUC": r["AUC"]})
    fig_r = plot_roc(probs, yt, class_names, class_rows)
    fig_r.savefig(out_dir / f"{excel_name}_{split}_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig_r)
    fig_c = plot_confusion(yt, yhat, class_names)
    fig_c.savefig(out_dir / f"{excel_name}_{split}_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig_c)


# ----------------------------------------------------------------------------
# Excel export (3 sheets: 总体指标 / 每类指标 / 排名)
# ----------------------------------------------------------------------------
def export_excel(out_xlsx: Path, summary_df: pd.DataFrame, per_class_df: pd.DataFrame) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    # 排名 sheet
    rank_rows: list[dict] = []
    for metric_col, label in (("auc_point", "auc"), ("acc_point", "acc")):
        pts = [
            (r["model"], r[metric_col])
            for _, r in summary_df.iterrows()
            if pd.notna(r.get(metric_col))
        ]
        pts.sort(key=lambda x: -x[1])
        for rank, (model, point) in enumerate(pts, start=1):
            raw = summary_df.loc[summary_df["model"] == model, label].iloc[0]
            rank_rows.append({"metric": label.upper(), "rank": rank, "model": model, "value": raw, "point": point})
    ranking = pd.DataFrame(rank_rows)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="总体指标", index=False)
        per_class_df.to_excel(writer, sheet_name="每类指标", index=False)
        ranking.to_excel(writer, sheet_name="排名", index=False)


# ----------------------------------------------------------------------------
# Package builder
# ----------------------------------------------------------------------------
def build_table_package(
    *,
    table_name: str,           # "table1" or "table2"
    out_dir: Path,
    pool_dir: Path,
    pool_suffix: str,          # "_test_pool_predictions.npz" / "_val_pool_predictions.npz"
    split: str,                # "test" / "val"
    split_dir: Path,
    ckpt_root_name: str,       # "v2" / "v3"
    plot_basename: str,        # "test" / "val" (filename prefix for plots)
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    caches_dir = out_dir / "caches"
    caches_dir.mkdir(parents=True, exist_ok=True)

    macro_rows: list[dict] = []
    pc_rows: list[dict] = []
    manifest_models: dict[str, dict] = {}

    split_total, split_counts = list_split_class_counts(split_dir)
    print(f"\n=== {table_name.upper()} ({split} n={split_total}, ckpt={ckpt_root_name}) ===")
    print(f"  per-class: {split_counts}")

    for model in MODELS:
        ck_name = CK_MAP[model]
        pool_path = pool_dir / f"{model}{pool_suffix}"
        if not pool_path.is_file():
            raise FileNotFoundError(f"Missing pool cache: {pool_path}")
        pool = load_pool(pool_path)
        if split == "test":
            probs, yt, yhat, class_names, idx = filter_test(pool)
        else:
            probs, yt, yhat, class_names, idx = filter_val(pool)
        n = len(yt)
        print(f"  [{model}] pool={pool_path.name} n_{split}={n}")

        # Save filtered cache (for reproducibility / re-use)
        filtered_cache = caches_dir / f"{model}_{split}_predictions.npz"
        save_cache(filtered_cache, probs, yt, yhat, class_names)

        # Metrics + per-class
        row, pc = compute_metrics_block(model, ck_name, split, yt, yhat, probs, class_names)
        row["class_counts"] = json.dumps(dict(Counter(yt.tolist())))
        # Normalize naming: model = excel name (e.g. "casgnet"), checkpoint = ck_name
        row["checkpoint"] = ck_name
        row["model"] = model  # overwrites row["model"]=ck_name from row_eltra_bootstrap
        macro_rows.append(row)
        pc_rows.extend(pc)

        # Plots
        make_plots(plots_dir, model, plot_basename, yt, yhat, probs, class_names)

        # Per-model artifacts
        mdir = per_model_dir / model
        mdir.mkdir(parents=True, exist_ok=True)
        overall = {k: row.get(k, "") for k in (
            "model", "checkpoint", "split", "n_samples", "class_counts",
            "auc", "sensitivity", "specificity", "npv", "ppv", "acc",
        )}
        (mdir / "metrics_overall.json").write_text(
            json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([overall]).to_csv(mdir / "metrics_overall.csv", index=False)

        # per-class rows for this model still carry class name in "model" column
        # and excel name in "excel_model"; normalize before saving.
        model_pc = pd.DataFrame([p for p in pc if p.get("excel_model") == model]).copy()
        if "experiment" in model_pc.columns:
            model_pc = model_pc.drop(columns=["experiment"], errors="ignore")
        if "model" in model_pc.columns:
            model_pc = model_pc.rename(columns={"model": "class"})
        if "excel_model" in model_pc.columns:
            model_pc = model_pc.rename(columns={"excel_model": "model"})
        model_pc.to_csv(mdir / "metrics_per_class.csv", index=False)

        # Copy plots with canonical names
        shutil.copy2(plots_dir / f"{model}_{plot_basename}_auc.png", mdir / f"{split}_roc.png")
        shutil.copy2(plots_dir / f"{model}_{plot_basename}_confusion.png", mdir / f"{split}_confusion.png")
        shutil.copy2(plots_dir / f"{model}_{plot_basename}_auc.csv", mdir / f"{split}_roc_auc.csv")

        manifest_models[model] = {
            "complete": True,
            "n_samples": n,
            "cache": str(filtered_cache.relative_to(out_dir)),
            "artifacts": [
                "metrics_overall.json", "metrics_overall.csv", "metrics_per_class.csv",
                f"{split}_roc.png", f"{split}_confusion.png", f"{split}_roc_auc.csv",
            ],
        }

    # Build macro dataframe. row["model"] is already the excel name (e.g. "casgnet");
    # row["checkpoint"] is the ck subdir name.
    macro_df = pd.DataFrame(macro_rows)
    macro_df["auc_point"] = macro_df["auc"].map(parse_point)
    macro_df["acc_point"] = macro_df["acc"].map(parse_point)
    macro_df["sensitivity_point"] = macro_df["sensitivity"].map(parse_point)
    macro_df["specificity_point"] = macro_df["specificity"].map(parse_point)
    macro_df["npv_point"] = macro_df["npv"].map(parse_point)
    macro_df["ppv_point"] = macro_df["ppv"].map(parse_point)
    macro_df = macro_df.sort_values("auc_point", ascending=False, kind="mergesort").reset_index(drop=True)
    macro_df["rank"] = range(1, len(macro_df) + 1)

    summary_cols = [
        "rank", "model", "checkpoint", "split", "n_samples", "class_counts",
        "acc", "auc", "sensitivity", "specificity", "npv", "ppv",
        "auc_point", "acc_point", "sensitivity_point", "specificity_point", "npv_point", "ppv_point",
    ]
    summary_df = macro_df[[c for c in summary_cols if c in macro_df.columns]].copy()
    summary_df.to_csv(out_dir / f"{table_name.upper()}_SUMMARY.csv", index=False)

    # Per-class csv
    pc_df = pd.DataFrame(pc_rows)
    # pc_rows from bootstrap_per_class_comparison_rows carry the class name in the
    # "model" column (PER_CLASS_COMPARISON_FIELDS). Rename so that:
    #   "model"   (class name) -> "class"
    #   "excel_model"          -> "model"   (the actual model name)
    if "experiment" in pc_df.columns:
        pc_df = pc_df.drop(columns=["experiment"], errors="ignore")
    if "model" in pc_df.columns:
        pc_df = pc_df.rename(columns={"model": "class"})
    if "excel_model" in pc_df.columns:
        pc_df = pc_df.rename(columns={"excel_model": "model"})
    pc_cols = ["model", "split", "class", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    pc_df = pc_df[[c for c in pc_cols if c in pc_df.columns]].reset_index(drop=True)
    # Order rows by model auc rank then class
    order_map = {m: i for i, m in enumerate(summary_df["model"].tolist())}
    pc_df["_ord"] = pc_df["model"].map(lambda m: order_map.get(m, 999)).astype(int)
    pc_df = pc_df.sort_values(["_ord", "class"], kind="mergesort").drop(columns=["_ord"]).reset_index(drop=True)
    pc_df.to_csv(out_dir / f"{table_name.upper()}_PER_CLASS.csv", index=False)

    # Excel
    export_excel(out_dir / f"{table_name.upper()}_RESULTS.xlsx", summary_df, pc_df)

    # Summary MD
    md = [f"# {table_name.upper()} Summary (ORIGINAL {split} split, n={split_total})",
          "",
          f"{ckpt_root_name} checkpoints + legacy_val_resize; **no subset search**; bootstrap n={N_BOOTSTRAP} seed={BOOTSTRAP_SEED}.",
          "",
          "| Rank | Model | n | ACC | AUC | SENS | SPEC | NPV | PPV |",
          "|------|-------|---|-----|-----|------|------|-----|-----|"]
    for _, r in summary_df.iterrows():
        md.append(
            f"| {r['rank']} | {r['model']} | {r['n_samples']} | {r['acc']} | {r['auc']} | "
            f"{r['sensitivity']} | {r['specificity']} | {r['npv']} | {r['ppv']} |"
        )
    (out_dir / f"{table_name.upper()}_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # Package manifest
    pkg_manifest = {
        "table": table_name,
        "split": split,
        "split_dir": str(split_dir.relative_to(ROOT)),
        "n_total": split_total,
        "per_class_counts": split_counts,
        "ckpt_root": ckpt_root_name,
        "ckpt_root_path": f"checkpoints/old_data_supcon_compare_{ckpt_root_name}",
        "preprocessing": "legacy_val_resize (Resize 224x224)",
        "bootstrap": {"n": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "confidence": 0.95},
        "subset_search": False,
        "models": manifest_models,
        "auc_ranking": [
            [r["model"], float(r["auc_point"])] for _, r in summary_df.iterrows()
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(pkg_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # README
    readme = f"""# {table_name.upper()} Final Package (ORIGINAL split)

**{table_name.upper()} {split} set** — ORIGINAL `{split_dir.relative_to(ROOT)}` (n={split_total}), {ckpt_root_name} checkpoints, legacy_val_resize.
**No subset search.** All images in the split are used.

## Contents

| File | Description |
|------|-------------|
| `{table_name.upper()}_SUMMARY.csv` / `.md` | 8 models, 6 metrics + 95% CI |
| `{table_name.upper()}_PER_CLASS.csv` | 56 rows (8 models x 7 classes) |
| `{table_name.upper()}_RESULTS.xlsx` | Excel: 总体指标 / 每类指标 / 排名 |
| `per_model/<model>/` | metrics + `{split}_roc.png` + `{split}_confusion.png` |
| `caches/<model>_{split}_predictions.npz` | Filtered prediction cache |
| `manifest.json` | Package manifest |

## Settings

- Bootstrap: n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}, 95% CI
- Preprocessing: Resize 224x224 (legacy_val_resize)
- Checkpoints: `checkpoints/old_data_supcon_compare_{ckpt_root_name}/*/best_auc_model.pth`

## AUC ranking

"""
    for _, r in summary_df.iterrows():
        readme += f"{int(r['rank'])}. {r['model']} — AUC {r['auc']}\n"
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "out_dir": str(out_dir.relative_to(HERE)),
        "n_total": split_total,
        "per_class_counts": split_counts,
        "summary_df": summary_df,
        "per_class_df": pc_df,
        "auc_ranking": pkg_manifest["auc_ranking"],
    }


def load_searched_summary(table: str) -> dict | None:
    """Load the searched (option_b) per-model macro metrics for before/after compare."""
    p = HERE / "option_b_snapshot" / "option_b_summary.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d


def build_snapshot(
    t1_result: dict, t2_result: dict,
    test_total: int, test_counts: dict[str, int],
    val_total: int, val_counts: dict[str, int],
) -> None:
    if SNAP_DIR.exists():
        shutil.rmtree(SNAP_DIR)
    SNAP_DIR.mkdir(parents=True)

    # Image lists
    (SNAP_DIR / "test_image_list.txt").write_text(
        "\n".join(collect_split_image_list(TEST_DIR)) + "\n", encoding="utf-8"
    )
    (SNAP_DIR / "val_image_list.txt").write_text(
        "\n".join(collect_split_image_list(VAL_DIR)) + "\n", encoding="utf-8"
    )

    # Per-split manifests
    test_manifest = {
        "split": "test", "n_total": test_total,
        "per_class_counts": test_counts,
        "source": str(TEST_DIR.relative_to(ROOT)),
    }
    val_manifest = {
        "split": "val", "n_total": val_total,
        "per_class_counts": val_counts,
        "source": str(VAL_DIR.relative_to(ROOT)),
    }
    (SNAP_DIR / "test_manifest.json").write_text(
        json.dumps(test_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (SNAP_DIR / "val_manifest.json").write_text(
        json.dumps(val_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Training val_auc reference (best_val_auc_on_save) for each ckpt
    train_val_auc: dict[str, dict] = {}
    for table, root_name in (("table1", "v2"), ("table2", "v3")):
        train_val_auc[table] = {}
        for model in MODELS:
            ck = CK_MAP[model]
            p = ROOT / f"checkpoints/old_data_supcon_compare_{root_name}" / ck / "result_summary.json"
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                train_val_auc[table][model] = {
                    "best_val_auc_on_save": d.get("best_val_auc_on_save"),
                    "reloaded_val_auc": d.get("reloaded_val_auc"),
                    "n_val": d.get("n_val"),
                }

    # Build summary JSON
    def _rank_dict(summary_df: pd.DataFrame) -> list[dict]:
        return [
            {
                "rank": int(r["rank"]),
                "model": r["model"],
                "n": int(r["n_samples"]),
                "auc": float(r["auc_point"]),
                "acc": float(r["acc_point"]),
                "sensitivity": float(r["sensitivity_point"]),
                "specificity": float(r["specificity_point"]),
                "npv": float(r["npv_point"]),
                "ppv": float(r["ppv_point"]),
                "auc_ci_str": r["auc"],
                "acc_ci_str": r["acc"],
            }
            for _, r in summary_df.iterrows()
        ]

    t1_summary = _rank_dict(t1_result["summary_df"])
    t2_summary = _rank_dict(t2_result["summary_df"])

    # T1 vs T2 comparison
    t1_by = {r["model"]: r for r in t1_summary}
    t2_by = {r["model"]: r for r in t2_summary}
    t1_vs_t2: dict[str, dict] = {}
    for m in MODELS:
        t1 = t1_by[m]; t2 = t2_by[m]
        t1_vs_t2[m] = {
            "t1_auc": t1["auc"], "t2_auc": t2["auc"],
            "t1_minus_t2_auc": t1["auc"] - t2["auc"],
            "t1_acc": t1["acc"], "t2_acc": t2["acc"],
            "t1_minus_t2_acc": t1["acc"] - t2["acc"],
            "t1_gt_t2_auc": t1["auc"] > t2["auc"],
            "t1_gt_t2_acc": t1["acc"] > t2["acc"],
        }

    # Test vs training val_auc consistency
    consistency: dict[str, dict] = {}
    for m in MODELS:
        t1_auc = t1_by[m]["auc"]
        t2_auc = t2_by[m]["auc"]
        v2_train = train_val_auc["table1"][m]["best_val_auc_on_save"]
        v3_train = train_val_auc["table2"][m]["best_val_auc_on_save"]
        consistency[m] = {
            "t1_test_auc": t1_auc,
            "t1_train_val_auc_v2": v2_train,
            "t1_test_le_train_val": (t1_auc <= v2_train) if v2_train is not None else None,
            "t1_test_minus_train_val": t1_auc - v2_train if v2_train is not None else None,
            "t2_val_auc": t2_auc,
            "t2_train_val_auc_v3": v3_train,
            "t2_val_le_train_val": (t2_auc <= v3_train) if v3_train is not None else None,
            "t2_val_minus_train_val": t2_auc - v3_train if v3_train is not None else None,
        }

    # Before/after vs searched version
    searched = load_searched_summary("option_b")
    before_after: dict = {}
    if searched:
        s_after = searched.get("after", {})
        for table in ("table1", "table2"):
            before_after[table] = {}
            for m in MODELS:
                s = s_after.get(table, {}).get(m, {})
                cur = (t1_by if table == "table1" else t2_by)[m]
                before_after[table][m] = {
                    "searched_auc": s.get("auc"),
                    "original_auc": cur["auc"],
                    "delta_auc": cur["auc"] - s.get("auc", 0) if s.get("auc") is not None else None,
                    "searched_acc": s.get("acc"),
                    "original_acc": cur["acc"],
                    "delta_acc": cur["acc"] - s.get("acc", 0) if s.get("acc") is not None else None,
                }

    summary = {
        "option": "B_final_original_split",
        "description": "Use ORIGINAL old_data/test (n=258) and old_data/val (n=207); no subset search.",
        "date": "2026-06-28",
        "models": MODELS,
        "splits": {
            "test": {"n_total": test_total, "per_class_counts": test_counts},
            "val": {"n_total": val_total, "per_class_counts": val_counts},
        },
        "t1_auc_ranking": t1_result["auc_ranking"],
        "t2_auc_ranking": t2_result["auc_ranking"],
        "t1_summary": t1_summary,
        "t2_summary": t2_summary,
        "t1_vs_t2": t1_vs_t2,
        "training_val_auc_reference": train_val_auc,
        "test_vs_train_val_consistency": consistency,
        "before_after_vs_searched": before_after,
        "packages": {
            "table1": t1_result["out_dir"],
            "table2": t2_result["out_dir"],
        },
        "snapshot_dir": str(SNAP_DIR.relative_to(HERE)),
        "files": {
            "test_image_list": "test_image_list.txt",
            "val_image_list": "val_image_list.txt",
            "test_manifest": "test_manifest.json",
            "val_manifest": "val_manifest.json",
        },
    }
    (SNAP_DIR / "original_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Before/after comparison CSV
    if before_after:
        rows = []
        for table in ("table1", "table2"):
            for m in MODELS:
                ba = before_after[table][m]
                rows.append({
                    "table": table, "model": m,
                    "searched_auc": ba["searched_auc"], "original_auc": ba["original_auc"],
                    "delta_auc": ba["delta_auc"],
                    "searched_acc": ba["searched_acc"], "original_acc": ba["original_acc"],
                    "delta_acc": ba["delta_acc"],
                })
        pd.DataFrame(rows).to_csv(SNAP_DIR / "before_after_vs_searched.csv", index=False)

    print(f"\nSnapshot: {SNAP_DIR}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    test_total, test_counts = list_split_class_counts(TEST_DIR)
    val_total, val_counts = list_split_class_counts(VAL_DIR)
    print(f"Original test split: n={test_total}, counts={test_counts}")
    print(f"Original val   split: n={val_total}, counts={val_counts}")

    t1 = build_table_package(
        table_name="table1",
        out_dir=T1_OUT,
        pool_dir=T1_POOL_DIR,
        pool_suffix="_test_pool_predictions.npz",
        split="test",
        split_dir=TEST_DIR,
        ckpt_root_name="v2",
        plot_basename="test",
    )
    t2 = build_table_package(
        table_name="table2",
        out_dir=T2_OUT,
        pool_dir=T2_POOL_DIR,
        pool_suffix="_val_pool_predictions.npz",
        split="val",
        split_dir=VAL_DIR,
        ckpt_root_name="v3",
        plot_basename="val",
    )

    build_snapshot(t1, t2, test_total, test_counts, val_total, val_counts)

    print("\n=== T1 AUC ranking ===")
    for r in t1["auc_ranking"]:
        print(f"  {r[0]:15s} {r[1]:.4f}")
    print("\n=== T2 AUC ranking ===")
    for r in t2["auc_ranking"]:
        print(f"  {r[0]:15s} {r[1]:.4f}")

    print(f"\nDone. T1: {T1_OUT}")
    print(f"      T2: {T2_OUT}")
    print(f"      Snapshot: {SNAP_DIR}")


if __name__ == "__main__":
    main()
