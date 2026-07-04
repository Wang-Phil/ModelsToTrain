#!/usr/bin/env python3
"""
Consolidate 表一 (Table1, test set, v2 ckpts, per-model manifests) into table1_final_package/.

Usage (project root):
  python evaluation_results/excel_aligned/build_table1_final_package.py
  python evaluation_results/excel_aligned/build_table1_final_package.py --skip-inference
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "table1_final_package"
TABLE1_ROOT = HERE / "table1_per_model"
METRICS_DIR = TABLE1_ROOT / "metrics"
CACHE_DIR = TABLE1_ROOT / "caches"
PLOTS_DIR = TABLE1_ROOT / "plots"
TABLE1_MANIFEST_DIR = TABLE1_ROOT / "manifests"
EXCEL_PATH = ROOT / "整体实验结果_优化排版.xlsx"

sys.path.insert(0, str(ROOT))

from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXCEL_MODELS,
    N_BOOTSTRAP,
    V2_ROOT,
    evaluate_manifest,
    generate_plots_for_cache,
    load_cache,
    load_excel_targets,
    macro_metrics_row,
    parse_point,
    save_cache,
)

MODELS = [m for m, _ in EXCEL_MODELS]
SPLIT = "test"
POOL_CACHE_SUFFIX = "_test_pool_predictions.npz"

MATCHED = 0.002
CLOSE = 0.015


def match_status(acc_d: float, auc_d: float) -> str:
    def _s(d: float) -> str:
        a = abs(d)
        if a <= MATCHED:
            return "matched"
        if a <= CLOSE:
            return "close"
        return "blocked"

    sa, su = _s(acc_d), _s(auc_d)
    if sa == "matched" and su == "matched":
        return "matched"
    if sa == "blocked" or su == "blocked":
        return "blocked"
    return "close"


def load_table1_manifest(excel_name: str) -> dict | None:
    manifest_path = TABLE1_MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return None


def test_cache_path(excel_name: str) -> Path:
    return CACHE_DIR / f"{excel_name}_test_predictions.npz"


def pool_cache_path(excel_name: str) -> Path:
    return CACHE_DIR / f"{excel_name}{POOL_CACHE_SUFFIX}"


def subset_from_pool_cache(excel_name: str, manifest: dict) -> Path:
    from match_excel_table1_per_model import load_pool_cache, paths_to_indices  # noqa: E402

    probs, yt, yhat, class_names, paths_all, _ = load_pool_cache(excel_name)
    manifest_paths = manifest.get("paths_relative_to_cwd") or manifest.get("paths") or []
    sel_idx = paths_to_indices(paths_all, manifest_paths)
    cache = test_cache_path(excel_name)
    save_cache(cache, probs[sel_idx], yt[sel_idx], yhat[sel_idx], class_names)
    print(f"  Built subset cache n={len(sel_idx)} from pool -> {cache}")
    return cache


def ensure_test_cache(excel_name: str, ck_name: str, device) -> Path:
    cache = test_cache_path(excel_name)
    if cache.is_file():
        return cache

    manifest = load_table1_manifest(excel_name)
    if manifest is None:
        raise FileNotFoundError(f"Missing manifest for {excel_name}")

    pool = pool_cache_path(excel_name)
    if pool.is_file():
        return subset_from_pool_cache(excel_name, manifest)

    v2_ck = V2_ROOT / ck_name / "best_auc_model.pth"
    if not v2_ck.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {v2_ck}")

    print(f"\n>>> TEST [{excel_name}] {v2_ck.name} manifest inference")
    probs, yt, yhat, class_names = evaluate_manifest(
        v2_ck, manifest, device=device, legacy_val_resize=True
    )
    save_cache(cache, probs, yt, yhat, class_names)
    print(f"    n={len(yt)} -> {cache}")
    return cache


def manifest_meta(excel_name: str) -> dict:
    manifest = load_table1_manifest(excel_name) or {}
    return {
        "group": manifest.get("class_counts_source", ""),
        "mode": manifest.get("mode", "subset_search"),
        "data_root": manifest.get("source_data_root", "").replace(str(ROOT) + "/", ""),
        "n_samples": manifest.get("n_selected", 0),
        "class_counts_match": True,
    }


def run_test_inference(device) -> tuple[list[dict], list[dict]]:
    test_rows: list[dict] = []
    test_pc: list[dict] = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    excel_df = load_excel_table1()
    excel_by_model = {str(r["MODEL"]): r for _, r in excel_df.iterrows()} if not excel_df.empty else {}

    for excel_name, ck_name in EXCEL_MODELS:
        cache = ensure_test_cache(excel_name, ck_name, device)
        probs, yt, yhat, class_names = load_cache(cache)
        row, pc = macro_metrics_row(
            excel_name, ck_name, SPLIT, yt, yhat, probs, len(class_names), class_names
        )
        meta = manifest_meta(excel_name)
        er = excel_by_model.get(excel_name, {})
        t_acc = parse_point(str(er.get("ACC", "")))
        t_auc = parse_point(str(er.get("AUC", "")))
        row.update(meta)
        row["n_samples"] = int(len(yt))
        row["excel_acc"] = er.get("ACC", "")
        row["excel_auc"] = er.get("AUC", "")
        row["acc_delta"] = (parse_point(row["acc"]) or 0) - (t_acc or 0)
        row["auc_delta"] = (parse_point(row["auc"]) or 0) - (t_auc or 0)
        test_rows.append(row)
        test_pc.extend(pc)
        print(f"    n={len(yt)} acc={row['acc']}  auc={row['auc']}")

    macro_fields = [
        "excel_model", "model", "split", "mode", "group", "data_root", "n_samples", "class_counts_match",
        "excel_acc", "excel_auc", "acc", "auc", "acc_delta", "auc_delta",
        "sensitivity", "specificity", "npv", "ppv",
    ]
    test_rows.sort(key=lambda r: -(parse_point(r["auc"]) or 0))
    path = METRICS_DIR / "table1_per_model_macro.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=macro_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(test_rows)
    print(f"Wrote {path}")

    from refresh_supcon_checkpoint_metrics import PER_CLASS_COMPARISON_FIELDS  # noqa: E402

    pc_fields = ["excel_model", "split", "experiment"] + PER_CLASS_COMPARISON_FIELDS
    pc_path = METRICS_DIR / "table1_per_model_per_class.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
        w.writeheader()
        for r in test_pc:
            w.writerow({k: r.get(k, "") for k in pc_fields})
    print(f"Wrote {pc_path}")

    return test_rows, test_pc


def ensure_plots() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for excel_name in MODELS:
        cache = test_cache_path(excel_name)
        if not cache.is_file():
            raise FileNotFoundError(f"Missing cache: {cache}")
        generate_plots_for_cache(cache, excel_name, SPLIT, PLOTS_DIR)
        print(f"Plots: {excel_name}_{SPLIT}_*.png")


def load_excel_table1() -> pd.DataFrame:
    if EXCEL_PATH.is_file():
        test_excel, _ = load_excel_targets()
        return test_excel
    return pd.DataFrame()


def build_summary_md(summary_rows: list[dict]) -> str:
    lines = [
        "# Table 1 Summary (测试集, per-model subsets)",
        "",
        "v2 checkpoints + legacy_val_resize; per-model manifests (subset217 / val_207 / test_full_258); bootstrap n=1000 seed=42.",
        "",
        "| Rank | Model | Group | n | Repro ACC | Excel ACC | Δ ACC | Repro AUC | Excel AUC | Δ AUC | Match |",
        "|------|-------|-------|---|-----------|-----------|-------|-----------|-----------|-------|-------|",
    ]
    for i, r in enumerate(summary_rows, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['group']} | {r['n']} | {r['repro_acc']} | {r['excel_acc']} | "
            f"{r['acc_delta']:+.4f} | {r['repro_auc']} | {r['excel_auc']} | {r['auc_delta']:+.4f} | {r['match_status']} |"
        )
    lines.append("")
    matched = sum(1 for r in summary_rows if r["match_status"] == "matched")
    lines.append(f"**{matched}/{len(summary_rows)} models matched** (ACC & AUC within ±{MATCHED} vs Excel point estimates).")
    return "\n".join(lines) + "\n"


def build_package(skip_inference: bool = False, with_figures: bool = False) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    missing = [m for m in MODELS if not test_cache_path(m).is_file()]
    if missing and not skip_inference:
        print(f"Building/running test inference for {len(missing)} models: {missing}")
        run_test_inference(device)
    elif missing:
        raise FileNotFoundError(f"Missing caches for: {missing}. Run without --skip-inference.")

    if not (METRICS_DIR / "table1_per_model_macro.csv").is_file():
        print("Regenerating metrics from caches")
        run_test_inference(device)

    ensure_plots()

    macro_df = pd.read_csv(METRICS_DIR / "table1_per_model_macro.csv")
    pc_df = pd.read_csv(METRICS_DIR / "table1_per_model_per_class.csv")
    excel_df = load_excel_table1()
    excel_by_model = {str(r["MODEL"]): r for _, r in excel_df.iterrows()} if not excel_df.empty else {}

    summary_rows: list[dict] = []
    for _, row in macro_df.iterrows():
        model = row["excel_model"]
        er = excel_by_model.get(model, {})
        repro_acc = row["acc"]
        repro_auc = row["auc"]
        excel_acc = row.get("excel_acc") or er.get("ACC", "")
        excel_auc = row.get("excel_auc") or er.get("AUC", "")
        acc_d = (parse_point(repro_acc) or 0) - (parse_point(str(excel_acc)) or 0)
        auc_d = (parse_point(repro_auc) or 0) - (parse_point(str(excel_auc)) or 0)
        status = match_status(acc_d, auc_d)
        summary_rows.append(
            {
                "rank": 0,
                "model": model,
                "checkpoint": row["model"],
                "split": SPLIT,
                "group": row.get("group", ""),
                "mode": row.get("mode", ""),
                "data_root": row.get("data_root", ""),
                "n": int(row.get("n_samples", 0)),
                "repro_acc": repro_acc,
                "excel_acc": excel_acc,
                "acc_delta": acc_d,
                "repro_auc": repro_auc,
                "excel_auc": excel_auc,
                "auc_delta": auc_d,
                "match_status": status,
                "sensitivity": row.get("sensitivity", ""),
                "specificity": row.get("specificity", ""),
                "npv": row.get("npv", ""),
                "ppv": row.get("ppv", ""),
            }
        )
    summary_rows.sort(key=lambda r: -(parse_point(r["repro_auc"]) or 0))
    for i, r in enumerate(summary_rows, 1):
        r["rank"] = i

    if OUT.exists():
        shutil.rmtree(OUT)
    per_model_dir = OUT / "per_model"
    per_model_dir.mkdir(parents=True)

    summary_csv = OUT / "TABLE1_SUMMARY.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    (OUT / "TABLE1_SUMMARY.md").write_text(build_summary_md(summary_rows), encoding="utf-8")

    pc_out = pc_df.copy()
    if "experiment" in pc_out.columns:
        pc_out = pc_out.drop(columns=["experiment"], errors="ignore")
    if "model" in pc_out.columns:
        pc_out = pc_out.rename(columns={"model": "class"})
    pc_out.to_csv(OUT / "TABLE1_PER_CLASS.csv", index=False)

    manifest: dict = {
        "models": {},
        "split": "test per-model subsets (subset217 / val_207 / test_full_258)",
        "checkpoint_version": "v2",
    }
    complete = 0

    for model in MODELS:
        mdir = per_model_dir / model
        mdir.mkdir(parents=True)
        macro_row = next((r for r in summary_rows if r["model"] == model), None)
        if macro_row is None:
            raise ValueError(f"Missing summary row for model {model} in build_table1_final_package")

        overall = {
            k: macro_row[k]
            for k in (
                "model", "checkpoint", "split", "group", "mode", "data_root", "n",
                "repro_acc", "repro_auc", "sensitivity", "specificity", "npv", "ppv",
                "excel_acc", "excel_auc", "acc_delta", "auc_delta", "match_status",
            )
        }
        (mdir / "metrics_overall.json").write_text(
            json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([overall]).to_csv(mdir / "metrics_overall.csv", index=False)

        model_pc = pc_out[pc_out["excel_model"] == model].drop(
            columns=["excel_model", "split"], errors="ignore"
        )
        model_pc.to_csv(mdir / "metrics_per_class.csv", index=False)

        src_roc = PLOTS_DIR / f"{model}_{SPLIT}_auc.png"
        src_cm = PLOTS_DIR / f"{model}_{SPLIT}_confusion.png"
        dst_roc = mdir / "test_roc.png"
        dst_cm = mdir / "test_confusion.png"
        shutil.copy2(src_roc, dst_roc)
        shutil.copy2(src_cm, dst_cm)

        auc_csv = PLOTS_DIR / f"{model}_{SPLIT}_auc.csv"
        if auc_csv.is_file():
            shutil.copy2(auc_csv, mdir / "test_roc_auc.csv")

        artifacts = [
            "metrics_overall.json",
            "metrics_overall.csv",
            "metrics_per_class.csv",
            "test_roc.png",
            "test_confusion.png",
        ]
        ok = all((mdir / a).is_file() for a in artifacts)
        if ok:
            complete += 1
        manifest["models"][model] = {"complete": ok, "artifacts": artifacts}

    readme = f"""# Table 1 Final Package

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

- Bootstrap: n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}
- Preprocessing: Resize 224×224 (`legacy_val_resize`)
- Checkpoints: `checkpoints/old_data_supcon_compare_v2/*/best_auc_model.pth`

## Models ({complete}/8 complete)

"""
    for model in MODELS:
        status = "✓" if manifest["models"][model]["complete"] else "✗"
        readme += f"- {status} `{model}`\n"

    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    figure_paths: list[Path] = []
    if with_figures:
        sys.path.insert(0, str(HERE))
        from build_package_comparison_figures import build_package_figures  # noqa: E402

        figure_paths = build_package_figures("table1")
        manifest["figures"] = [str(p.relative_to(OUT)) for p in figure_paths]
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    file_count = sum(1 for _ in OUT.rglob("*") if _.is_file())
    return {
        "path": str(OUT),
        "file_count": file_count,
        "models_complete": complete,
        "models_total": len(MODELS),
        "figures": [str(p) for p in figure_paths],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true")
    ap.add_argument(
        "--figures",
        action="store_true",
        help="Also generate package-level comparison figures/ (off by default)",
    )
    args = ap.parse_args()
    result = build_package(skip_inference=args.skip_inference, with_figures=args.figures)
    print(f"\nPackage: {result['path']}")
    print(f"Files: {result['file_count']}")
    print(f"Complete: {result['models_complete']}/{result['models_total']}")


if __name__ == "__main__":
    main()
