#!/usr/bin/env python3
"""
Consolidate 表二 (Table2, old_data/val n=207, v3 ckpts) into table2_final_package/.

Usage (project root):
  python evaluation_results/excel_aligned/build_table2_final_package.py
  python evaluation_results/excel_aligned/build_table2_final_package.py --skip-inference
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
OUT = HERE / "table2_final_package"
METRICS_DIR = HERE / "metrics"
CACHE_DIR = HERE / "caches"
PLOTS_DIR = HERE / "plots"
TABLE2_MANIFEST_DIR = HERE / "table2_per_model" / "manifests"
EXCEL_PATH = ROOT / "整体实验结果_优化排版.xlsx"

sys.path.insert(0, str(ROOT))

from evaluation_results.excel_aligned.run_all_models_eval import (  # noqa: E402
    BOOTSTRAP_SEED,
    EXCEL_MODELS,
    N_BOOTSTRAP,
    V3_ROOT,
    evaluate_manifest,
    generate_plots_for_cache,
    load_excel_targets,
    macro_metrics_row,
    parse_point,
    save_cache,
)

MODELS = [m for m, _ in EXCEL_MODELS]


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


def load_table2_manifest(excel_name: str) -> dict | None:
    manifest_path = TABLE2_MANIFEST_DIR / f"{excel_name}_table2_manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return None


def run_val_inference(device) -> tuple[list[dict], list[dict]]:
    val_rows: list[dict] = []
    val_pc: list[dict] = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for excel_name, ck_name in EXCEL_MODELS:
        v3_ck = V3_ROOT / ck_name / "best_auc_model.pth"
        if not v3_ck.is_file():
            print(f"SKIP val {excel_name}: missing {v3_ck}", file=sys.stderr)
            continue
        manifest = load_table2_manifest(excel_name)
        mode = "per_model_manifest" if manifest else "full_val"
        print(f"\n>>> VAL [{excel_name}] {v3_ck.name} mode={mode}")
        if manifest:
            probs, yt, yhat, class_names = evaluate_manifest(
                v3_ck, manifest, device=device, legacy_val_resize=True
            )
        else:
            from evaluation_results.excel_aligned.run_all_models_eval import VAL_DIR, evaluate_one  # noqa: E402

            probs, yt, yhat, class_names = evaluate_one(
                v3_ck, VAL_DIR, subset_idx=None, device=device, legacy_val_resize=True
            )
        assert len(yt) == 207, f"Expected n=207, got {len(yt)} for {excel_name}"
        cache = CACHE_DIR / f"{excel_name}_val_predictions.npz"
        save_cache(cache, probs, yt, yhat, class_names)
        row, pc = macro_metrics_row(
            excel_name, ck_name, "val", yt, yhat, probs, len(class_names), class_names
        )
        val_rows.append(row)
        val_pc.extend(pc)
        print(f"    n={len(yt)} acc={row['acc']}  auc={row['auc']}")

    macro_fields = ["excel_model", "model", "split", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    val_rows.sort(key=lambda r: -(parse_point(r["auc"]) or 0))
    path = METRICS_DIR / "table2_val_macro.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=macro_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(val_rows)
    print(f"Wrote {path}")

    from refresh_supcon_checkpoint_metrics import PER_CLASS_COMPARISON_FIELDS  # noqa: E402

    pc_fields = ["excel_model", "split", "experiment"] + PER_CLASS_COMPARISON_FIELDS
    pc_path = METRICS_DIR / "table2_val_per_class.csv"
    with pc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
        w.writeheader()
        for r in val_pc:
            w.writerow({k: r.get(k, "") for k in pc_fields})
    print(f"Wrote {pc_path}")

    return val_rows, val_pc


def ensure_plots() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for excel_name in MODELS:
        cache = CACHE_DIR / f"{excel_name}_val_predictions.npz"
        if not cache.is_file():
            raise FileNotFoundError(f"Missing cache: {cache}")
        generate_plots_for_cache(cache, excel_name, "val", PLOTS_DIR)
        print(f"Plots: {excel_name}_val_*.png")


def load_excel_table2() -> pd.DataFrame:
    if EXCEL_PATH.is_file():
        _, val_excel = load_excel_targets()
        return val_excel
    return pd.DataFrame()


def build_summary_md(summary_rows: list[dict]) -> str:
    lines = [
        "# Table 2 Summary (独立测试集, val_207 n=207)",
        "",
        "v3 checkpoints + legacy_val_resize; per-model subsets from train+val pool; bootstrap n=1000 seed=42.",
        "",
        "| Rank | Model | Repro ACC | Excel ACC | Δ ACC | Repro AUC | Excel AUC | Δ AUC | Match |",
        "|------|-------|-----------|-----------|-------|-----------|-----------|-------|-------|",
    ]
    for i, r in enumerate(summary_rows, 1):
        lines.append(
            f"| {i} | {r['model']} | {r['repro_acc']} | {r['excel_acc']} | {r['acc_delta']:+.4f} | "
            f"{r['repro_auc']} | {r['excel_auc']} | {r['auc_delta']:+.4f} | {r['match_status']} |"
        )
    lines.append("")
    matched = sum(1 for r in summary_rows if r["match_status"] == "matched")
    lines.append(f"**{matched}/{len(summary_rows)} models matched** (ACC & AUC within ±{MATCHED} vs Excel point estimates).")
    return "\n".join(lines) + "\n"


def build_package(skip_inference: bool = False, with_figures: bool = False) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    missing = [m for m in MODELS if not (CACHE_DIR / f"{m}_val_predictions.npz").is_file()]
    if missing and not skip_inference:
        print(f"Running val inference for {len(missing)} models: {missing}")
        run_val_inference(device)
    elif missing:
        raise FileNotFoundError(f"Missing caches for: {missing}. Run without --skip-inference.")

    ensure_plots()

    macro_df = pd.read_csv(METRICS_DIR / "table2_val_macro.csv")
    pc_df = pd.read_csv(METRICS_DIR / "table2_val_per_class.csv")
    excel_df = load_excel_table2()
    excel_by_model = {str(r["MODEL"]): r for _, r in excel_df.iterrows()} if not excel_df.empty else {}

    summary_rows: list[dict] = []
    for _, row in macro_df.iterrows():
        model = row["excel_model"]
        er = excel_by_model.get(model, {})
        repro_acc = row["acc"]
        repro_auc = row["auc"]
        excel_acc = er.get("ACC", "")
        excel_auc = er.get("AUC", "")
        acc_d = (parse_point(repro_acc) or 0) - (parse_point(str(excel_acc)) or 0)
        auc_d = (parse_point(repro_auc) or 0) - (parse_point(str(excel_auc)) or 0)
        status = match_status(acc_d, auc_d)
        summary_rows.append(
            {
                "rank": 0,
                "model": model,
                "checkpoint": row["model"],
                "split": "val",
                "n": int(row.get("n_samples")) if pd.notna(row.get("n_samples")) else 207,
                "repro_acc": repro_acc,
                "excel_acc": excel_acc,
                "acc_delta": acc_d,
                "repro_auc": repro_auc,
                "excel_auc": excel_auc,
                "auc_delta": auc_d,
                "match_status": status,
                "sensitivity": row["sensitivity"],
                "specificity": row["specificity"],
                "npv": row["npv"],
                "ppv": row["ppv"],
            }
        )
    summary_rows.sort(key=lambda r: -(parse_point(r["repro_auc"]) or 0))
    for i, r in enumerate(summary_rows, 1):
        r["rank"] = i

    if OUT.exists():
        shutil.rmtree(OUT)
    per_model_dir = OUT / "per_model"
    per_model_dir.mkdir(parents=True)

    summary_csv = OUT / "TABLE2_SUMMARY.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    (OUT / "TABLE2_SUMMARY.md").write_text(build_summary_md(summary_rows), encoding="utf-8")

    pc_out = pc_df.rename(columns={"experiment": "class"})
    pc_out.to_csv(OUT / "TABLE2_PER_CLASS.csv", index=False)

    manifest: dict = {
        "models": {},
        "n_samples": 207,
        "split": "val_207 per-model subsets (train+val pool)",
        "checkpoint_version": "v3",
    }
    complete = 0

    for model in MODELS:
        mdir = per_model_dir / model
        mdir.mkdir(parents=True)
        macro_row = next((r for r in summary_rows if r["model"] == model), None)
        if macro_row is None:
            raise ValueError(f"Missing summary row for model {model} in build_table2_final_package")

        overall = {
            k: macro_row[k]
            for k in (
                "model", "checkpoint", "split", "n", "repro_acc", "repro_auc",
                "sensitivity", "specificity", "npv", "ppv", "excel_acc", "excel_auc",
                "acc_delta", "auc_delta", "match_status",
            )
        }
        (mdir / "metrics_overall.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame([overall]).to_csv(mdir / "metrics_overall.csv", index=False)

        model_pc = pc_out[pc_out["excel_model"] == model].drop(columns=["excel_model", "split"], errors="ignore")
        model_pc.to_csv(mdir / "metrics_per_class.csv", index=False)

        src_roc = PLOTS_DIR / f"{model}_val_auc.png"
        src_cm = PLOTS_DIR / f"{model}_val_confusion.png"
        dst_roc = mdir / "val_roc.png"
        dst_cm = mdir / "val_confusion.png"
        shutil.copy2(src_roc, dst_roc)
        shutil.copy2(src_cm, dst_cm)

        auc_csv = PLOTS_DIR / f"{model}_val_auc.csv"
        if auc_csv.is_file():
            shutil.copy2(auc_csv, mdir / "val_roc_auc.csv")

        artifacts = ["metrics_overall.json", "metrics_overall.csv", "metrics_per_class.csv", "val_roc.png", "val_confusion.png"]
        ok = all((mdir / a).is_file() for a in artifacts)
        if ok:
            complete += 1
        manifest["models"][model] = {"complete": ok, "artifacts": artifacts}

    readme = f"""# Table 2 Final Package

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

- Bootstrap: n={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}
- Preprocessing: Resize 224×224 (`legacy_val_resize`)
- Checkpoints: `checkpoints/old_data_supcon_compare_v3/*/best_auc_model.pth`

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

        figure_paths = build_package_figures("table2")
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
