#!/usr/bin/env python3
"""
casgnet_s1_ce_only：用 --legacy-val-resize（Resize 224×224）复现 comparison 表中的
macro bootstrap 指标，并输出 ROC / 混淆矩阵到 evaluation_results/casgnet_metrics_rerun/。

用法（项目根）:
  python rerun_casgnet_legacy_metrics.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from compare_models_on_eltra_test import row_eltra_bootstrap, run_one_checkpoint
from eval_test_subset_bootstrap import manifest_paths_to_indices
from refresh_supcon_checkpoint_metrics import (
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
    infer_project_root,
)
from train_casgnet_contrastive_newdata import compute_macro_classification_metrics
from train_multiclass import ImageFolderDataset


def _eval_split(
    ck_path: Path,
    test_root: Path,
    *,
    subset_idx: np.ndarray | None,
    device,
    n_bootstrap: int,
    seed: int,
    legacy_val_resize: bool,
) -> dict:
    probs, yt, yhat, n_cls, class_names = run_one_checkpoint(
        ck_path,
        test_root,
        device=device,
        augmentation="standard",
        img_size=224,
        batch_size=32,
        num_workers=4,
        legacy_val_resize=legacy_val_resize,
    )
    if subset_idx is not None:
        yt = yt[subset_idx]
        yhat = yhat[subset_idx]
        probs = probs[subset_idx]

    row = row_eltra_bootstrap(
        "casgnet_s1_ce_only",
        yt,
        yhat,
        probs,
        n_cls,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    detail = row.pop("_detail")
    row.pop("_auc_sort", None)

    _, per_class_pt = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    aucs_ovr_pt = _per_class_auc_ovr(yt, probs, n_cls)
    per_class = bootstrap_per_class_comparison_rows(
        yt,
        yhat,
        probs,
        per_class_pt,
        aucs_ovr_pt,
        class_names,
        n_boot=n_bootstrap,
        random_state=seed,
    )
    detail["per_class"] = per_class
    detail["class_names"] = class_names
    return {"macro": row, "detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser(description="legacy resize 复现 casgnet_s1_ce_only bootstrap 指标 + 出图")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/old_data_supcon_compare_v2/casgnet_s1_ce_only/best_auc_model.pth"),
    )
    ap.add_argument("--test-dir", type=Path, default=Path("old_data/test"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("checkpoints/old_data_supcon_compare_v2/test_subset_table1_excel_aligned_manifest.json"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("evaluation_results/casgnet_metrics_rerun"),
    )
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--legacy-val-resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开启：Resize((224,224)) 复现 v2 表（默认: true）",
    )
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    import torch

    ck_path = args.checkpoint.resolve()
    if not ck_path.is_file():
        print(f"找不到 checkpoint: {ck_path}", file=sys.stderr)
        sys.exit(1)

    proj = infer_project_root(Path("checkpoints/old_data_supcon_compare_v2"))
    test_root = args.test_dir.resolve() if args.test_dir.is_absolute() else (proj / args.test_dir).resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} legacy_val_resize={args.legacy_val_resize}")

    print("\n=== full test bootstrap ===")
    full = _eval_split(
        ck_path,
        test_root,
        subset_idx=None,
        device=device,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        legacy_val_resize=args.legacy_val_resize,
    )
    print(json.dumps(full["macro"], indent=2, ensure_ascii=False))

    subset_idx = None
    if args.manifest.is_file():
        manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
        base = ImageFolderDataset(str(test_root), transform=None)
        subset_idx = manifest_paths_to_indices(manifest, test_root, base.samples)
        print(f"\n=== test subset bootstrap (n={len(subset_idx)}) ===")
        subset = _eval_split(
            ck_path,
            test_root,
            subset_idx=subset_idx,
            device=device,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            legacy_val_resize=args.legacy_val_resize,
        )
        print(json.dumps(subset["macro"], indent=2, ensure_ascii=False))
    else:
        subset = None
        print(f"\n跳过 subset（无 manifest: {args.manifest}）")

    # 写 CSV / JSON
    for tag, block in [("legacy_full_test", full), ("legacy_test_subset", subset)]:
        if block is None:
            continue
        macro_csv = out_dir / f"{tag}_macro.csv"
        with macro_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(block["macro"].keys()))
            w.writeheader()
            w.writerow(block["macro"])
        print(f"已写入: {macro_csv}")

        pc_csv = out_dir / f"{tag}_per_class.csv"
        pc_fields = ["model"] + PER_CLASS_COMPARISON_FIELDS
        with pc_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pc_fields)
            w.writeheader()
            for r in block["detail"]["per_class"]:
                w.writerow({"model": r.get("model", ""), **{k: r.get(k, "") for k in PER_CLASS_COMPARISON_FIELDS}})
        print(f"已写入: {pc_csv}")

        detail_json = out_dir / f"{tag}_detail.json"
        detail_json.write_text(json.dumps(block["detail"], indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入: {detail_json}")

    summary = {
        "checkpoint": str(ck_path),
        "legacy_val_resize": args.legacy_val_resize,
        "full_test": full["macro"],
        "test_subset": subset["macro"] if subset else None,
        "archived_reference": {
            "full_test_auc": "0.962(0.947-0.976)",
            "full_test_acc": "0.938(0.922-0.953)",
            "subset_auc": "0.970(0.954-0.983)",
            "subset_acc": "0.949(0.934-0.963)",
        },
    }
    summary_path = out_dir / "legacy_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入: {summary_path}")

    if not args.skip_plots:
        import subprocess

        plot_cmd = [
            sys.executable,
            str(proj / "plot_casgnet_val_eltra_roc_confusion.py"),
            "--checkpoint",
            str(ck_path),
            "--test-dir",
            str(test_root),
            "--out-dir",
            str(out_dir),
            "--skip-val",
        ]
        if args.legacy_val_resize:
            plot_cmd.append("--legacy-val-resize")
        print("\n>>> plots: full test")
        subprocess.run(plot_cmd + ["--plot-prefix", "casgnet_s1_ce_only_old_data_test"], check=True)

        if args.manifest.is_file():
            print("\n>>> plots: test subset")
            subprocess.run(
                plot_cmd
                + [
                    "--manifest",
                    str(args.manifest.resolve()),
                    "--plot-prefix",
                    "casgnet_s1_ce_only_test_subset",
                ],
                check=True,
            )


if __name__ == "__main__":
    main()
