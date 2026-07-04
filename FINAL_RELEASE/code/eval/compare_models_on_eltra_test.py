#!/usr/bin/env python3
"""
读取 comparison_summary.csv 中的 model 列，对每个实验目录的 best_auc_model.pth 在 eltra_test（ImageFolder）上评估，
macro-OVR AUC + macro OvR 分类指标，bootstrap 1000 次得到 95% CI，写出与 comparison_summary.csv 相同列格式的表格；
并按 validation 侧 comparison_summary.xlsx 相同结构写出第二张「各类 one-vs-rest」表（experiment + 列与 CSV 一致）。

用法（在项目根目录）:
  python compare_models_on_eltra_test.py \\
    --comparison-root checkpoints/old_data_supcon_compare \\
    --test-dir eltra_test \\
    --output-csv checkpoints/old_data_supcon_compare/comparison_summary_eltra_test.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from refresh_supcon_checkpoint_metrics import (
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
    fmt_ci,
    infer_project_root,
    load_comparison_dir_map,
    resolve_experiment_dir,
)
from train_casgnet_contrastive_newdata import (
    SupConClassifierNet,
    TransformSubset,
    bootstrap_auc_ci,
    bootstrap_classification_metrics_ci,
    collect_val_probs,
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)
from train_multiclass import ImageFolderDataset, get_data_augmentation


def _infer_proj_dims(state_dict: dict) -> tuple[int, int]:
    w0 = state_dict.get("proj.0.weight")
    w2 = state_dict.get("proj.2.weight")
    if w0 is None or w2 is None:
        return 128, 512
    return int(w2.shape[0]), int(w0.shape[0])


def run_one_checkpoint(
    ck_path: Path,
    test_root: Path,
    *,
    device: torch.device,
    augmentation: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    legacy_val_resize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    try:
        ckpt = torch.load(ck_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ck_path, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt)
    num_classes = int(ckpt.get("num_classes"))
    ck_class_to_idx: dict[str, int] = ckpt.get("class_to_idx") or {}
    class_names = [""] * num_classes
    for name, idx in ck_class_to_idx.items():
        class_names[int(idx)] = name
    if any(not n for n in class_names):
        raise ValueError(f"{ck_path}: class_to_idx 无法填满 0..num_classes-1")
    model_name = str(ckpt.get("model", "casgnet_s1"))
    proj_dim, hidden_dim = _infer_proj_dims(state_dict)

    _, val_aug = get_data_augmentation(
        augmentation_type=augmentation,
        img_size=img_size,
        legacy_val_resize=legacy_val_resize,
    )
    base = ImageFolderDataset(str(test_root), transform=None)

    if ck_class_to_idx and base.class_to_idx != ck_class_to_idx:
        raise ValueError(
            f"{ck_path}: class_to_idx 与 {test_root} 不一致。\n"
            f"  ckpt: {ck_class_to_idx}\n  test: {base.class_to_idx}"
        )

    n = len(base)
    subset = TransformSubset(base, np.arange(n), transform=val_aug)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    net = SupConClassifierNet(
        num_classes=num_classes,
        model_name=model_name,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        pretrained=False,
    )
    net.load_state_dict(state_dict, strict=True)
    net = net.to(device)

    use_amp = device.type == "cuda"
    probs, yt, yhat = collect_val_probs(net, loader, device, use_amp)
    return probs, yt, yhat, num_classes, class_names


def row_eltra_bootstrap(
    run_name: str,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    point_auc = compute_macro_auc_ovr(yt, probs)
    mean_b, lo, hi = bootstrap_auc_ci(yt, probs, n_boot=n_bootstrap, random_state=seed)
    cls_boot = bootstrap_classification_metrics_ci(
        yt, yhat, n_classes=n_classes, n_boot=n_bootstrap, random_state=seed
    )

    def _cell(key: str) -> tuple[float | None, float | None, float | None]:
        d = cls_boot.get(key, {}) or {}
        return (
            d.get("mean"),
            d.get("ci95_low"),
            d.get("ci95_high"),
        )

    macro_pt, _ = compute_macro_classification_metrics(yt, yhat, n_classes=n_classes)

    m_s, lo_s, hi_s = _cell("sensitivity")
    m_sp, lo_sp, hi_sp = _cell("specificity")
    m_npv, lo_npv, hi_npv = _cell("npv")
    m_ppv, lo_ppv, hi_ppv = _cell("ppv")
    m_acc, lo_acc, hi_acc = _cell("acc")

    return {
        "model": run_name,
        "auc": fmt_ci(mean_b, lo, hi),
        "sensitivity": fmt_ci(m_s, lo_s, hi_s),
        "specificity": fmt_ci(m_sp, lo_sp, hi_sp),
        "npv": fmt_ci(m_npv, lo_npv, hi_npv),
        "ppv": fmt_ci(m_ppv, lo_ppv, hi_ppv),
        "acc": fmt_ci(m_acc, lo_acc, hi_acc),
        "_auc_sort": point_auc if point_auc is not None else -1.0,
        "_detail": {
            "point_auc_macro_ovr": point_auc,
            "point_sensitivity": macro_pt["sensitivity"],
            "point_specificity": macro_pt["specificity"],
            "point_npv": macro_pt["npv"],
            "point_ppv": macro_pt["ppv"],
            "point_acc_macro_ovr": macro_pt["acc"],
            "accuracy_multiclass": float(np.mean(yt == yhat)),
            "n_samples": int(len(yt)),
            "bootstrap_n": n_bootstrap,
        },
    }


def read_models_column(csv_path: Path) -> list[str]:
    names: list[str] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = (row.get("model") or "").strip()
            if m:
                names.append(m)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="在 eltra_test 上复现 comparison_summary 风格的模型对比表")
    ap.add_argument("--comparison-root", type=Path, default=Path("checkpoints/old_data_supcon_compare"))
    ap.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="默认: <comparison-root>/comparison_summary.csv",
    )
    ap.add_argument("--test-dir", type=Path, default=Path("eltra_test"))
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="默认: <comparison-root>/comparison_summary_eltra_test.csv",
    )
    ap.add_argument("--output-json", type=Path, default=None, help="可选：逐模型 point 指标 JSON")
    ap.add_argument("--project-root", type=Path, default=None)
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--legacy-val-resize",
        action="store_true",
        help="验证/测试使用 Resize((img_size,img_size))，复现 old_data_supcon_compare_v2 指标",
    )
    args = ap.parse_args()

    comp_root = args.comparison_root.resolve()
    csv_in = (args.comparison_csv or (comp_root / "comparison_summary.csv")).resolve()
    proj = args.project_root.resolve() if args.project_root else infer_project_root(comp_root)
    td = args.test_dir
    test_root = td.resolve() if td.is_absolute() else (proj / td).resolve()
    out_csv = (args.output_csv or (comp_root / "comparison_summary_eltra_test.csv")).resolve()

    if not csv_in.is_file():
        print(f"找不到: {csv_in}", file=sys.stderr)
        sys.exit(1)
    if not test_root.is_dir():
        print(f"找不到测试目录: {test_root}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_map = load_comparison_dir_map(comp_root)
    names = read_models_column(csv_in)

    fieldnames = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    rows: list[dict] = []
    per_class_stacked: list[dict] = []
    details: dict[str, dict] = {}
    failures: list[str] = []

    for name in names:
        exp_dir = resolve_experiment_dir(comp_root, name, dir_map)
        if exp_dir is None:
            failures.append(f"{name}: 未找到实验目录")
            continue
        ck_path = exp_dir / "best_auc_model.pth"
        if not ck_path.is_file():
            failures.append(f"{name}: 缺少 {ck_path}")
            continue
        print(f"\n>>> [{name}] {ck_path}")
        try:
            probs, yt, yhat, n_cls, class_names = run_one_checkpoint(
                ck_path,
                test_root,
                device=device,
                augmentation=args.augmentation,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                legacy_val_resize=args.legacy_val_resize,
            )
        except Exception as e:
            failures.append(f"{name}: {e}")
            continue

        _, per_class_pt = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
        aucs_ovr_pt = _per_class_auc_ovr(yt, probs, n_cls)
        pc_rows = bootstrap_per_class_comparison_rows(
            yt,
            yhat,
            probs,
            per_class_pt,
            aucs_ovr_pt,
            class_names,
            n_boot=args.n_bootstrap,
            random_state=args.seed,
        )
        for sr in pc_rows:
            per_class_stacked.append({"experiment": name, **sr})

        row = row_eltra_bootstrap(
            name,
            yt,
            yhat,
            probs,
            n_cls,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
        det = row.pop("_detail")
        det["per_class_val"] = pc_rows
        details[name] = det
        rows.append(row)

    if failures:
        print("\n下列模型未完成评估:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)

    rows.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
    for r in rows:
        r.pop("_auc_sort", None)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写入: {out_csv}")

    detail_path = args.output_json.resolve() if args.output_json else out_csv.with_name(out_csv.stem + "_detail.json")
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入: {detail_path}")

    pc_csv = out_csv.with_name(out_csv.stem + "_per_class.csv")
    pc_fieldnames = ["experiment"] + PER_CLASS_COMPARISON_FIELDS
    with pc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_fieldnames)
        w.writeheader()
        for r in per_class_stacked:
            w.writerow({k: r.get(k, "") for k in pc_fieldnames})
    print(f"已写入: {pc_csv}")

    xlsx_path = out_csv.with_suffix(".xlsx")
    try:
        macro_df = pd.DataFrame(rows)
        pc_df = pd.DataFrame(per_class_stacked, columns=pc_fieldnames) if per_class_stacked else pd.DataFrame(columns=pc_fieldnames)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            macro_df.to_excel(writer, sheet_name="comparison_summary", index=False)
            pc_df.to_excel(writer, sheet_name="per_class_val", index=False)
        print(f"已写入: {xlsx_path}（工作表 comparison_summary + per_class_val）")
    except ModuleNotFoundError as e:
        if "openpyxl" in str(e):
            print("未安装 openpyxl，跳过 xlsx")
        else:
            raise


if __name__ == "__main__":
    main()
