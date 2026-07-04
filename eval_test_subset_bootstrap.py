#!/usr/bin/env python3
"""
在 test_subset_ranked_cas_first_manifest.json 指定的子集上，对各模型 best_auc_model.pth 评估：
macro-OVR AUC + macro OvR 分类指标，bootstrap（默认 1000）95% CI；
输出格式与 comparison_summary_test.csv 一致。

用法（项目根目录）:
  python eval_test_subset_bootstrap.py \\
    --comparison-root checkpoints/old_data_supcon_compare_v2

输出:
  - comparison_summary_test_subset.csv（宏观，含 bootstrap CI）
  - comparison_summary_test_subset_per_class.csv（experiment + 各类 one-vs-rest）
  - comparison_summary_test_subset.xlsx（若已安装 openpyxl）
  - comparison_summary_test_subset_point.csv（子集点估计，便于核对）
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

from compare_models_on_eltra_test import read_models_column, row_eltra_bootstrap
from compare_models_on_eltra_test import run_one_checkpoint as _run_one_checkpoint_base
from refresh_supcon_checkpoint_metrics import (
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
    infer_project_root,
    load_comparison_dir_map,
    resolve_experiment_dir,
)
from train_casgnet_contrastive_newdata import (
    SupConClassifierNet,
    collect_val_probs,
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)
from train_multiclass import ImageFolderDataset, get_data_augmentation


def run_one_checkpoint(ck_path: Path, test_root: Path, **kwargs):
    """与 compare_models_on_eltra_test 相同，但对 CASGNet 等旧权重允许 strict=False。"""
    try:
        return _run_one_checkpoint_base(ck_path, test_root, **kwargs)
    except RuntimeError as e:
        if "loading state_dict" not in str(e):
            raise
    device = kwargs["device"]
    augmentation = kwargs.get("augmentation", "standard")
    img_size = kwargs.get("img_size", 224)
    batch_size = kwargs.get("batch_size", 32)
    num_workers = kwargs.get("num_workers", 4)
    legacy_val_resize = kwargs.get("legacy_val_resize", False)

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
    model_name = str(ckpt.get("model") or "casgnet_s1")
    if not model_name and ckpt.get("model_variant"):
        model_name = f"casgnet_{ckpt.get('model_variant')}"

    from compare_models_on_eltra_test import TransformSubset, _infer_proj_dims

    proj_dim, hidden_dim = _infer_proj_dims(state_dict)
    _, val_aug = get_data_augmentation(
        augmentation_type=augmentation,
        img_size=img_size,
        legacy_val_resize=legacy_val_resize,
    )
    base = ImageFolderDataset(str(test_root), transform=None)
    if ck_class_to_idx and base.class_to_idx != ck_class_to_idx:
        raise ValueError(f"{ck_path}: class_to_idx 与 {test_root} 不一致")

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
    net.load_state_dict(state_dict, strict=False)
    net = net.to(device)
    use_amp = device.type == "cuda"
    probs, yt, yhat = collect_val_probs(net, loader, device, use_amp)
    print(f"  [warn] {ck_path.name}: load_state_dict strict=False（忽略旧版 GRN 等冗余键）")
    return probs, yt, yhat, num_classes, class_names


def _norm_path(p: str | Path) -> str:
    return str(Path(p).resolve().as_posix())


def manifest_paths_to_indices(manifest: dict, test_root: Path, base_samples: list[tuple[str, int]]) -> np.ndarray:
    """将 manifest 中的绝对路径映射为 ImageFolder 样本下标。"""
    path_to_idx: dict[str, int] = {}
    for i, (p, _lb) in enumerate(base_samples):
        path_to_idx[_norm_path(p)] = i

    rel_paths = manifest.get("paths_relative_to_cwd") or manifest.get("paths") or []
    if not rel_paths:
        raise ValueError("manifest 缺少 paths_relative_to_cwd")

    indices: list[int] = []
    missing: list[str] = []
    for raw in rel_paths:
        key = _norm_path(raw)
        if key not in path_to_idx:
            # 兼容 manifest 里写的是 test_root 下相对路径
            alt = _norm_path(test_root / Path(raw).name)
            if alt in path_to_idx:
                indices.append(path_to_idx[alt])
                continue
            missing.append(str(raw))
            continue
        indices.append(path_to_idx[key])

    if missing:
        raise ValueError(f"manifest 中有 {len(missing)} 条路径无法在 {test_root} 的 ImageFolder 中匹配，例如: {missing[0]}")

    idx = np.asarray(indices, dtype=np.int64)
    if idx.size != int(manifest.get("n_selected", idx.size)):
        print(
            f"警告: manifest n_selected={manifest.get('n_selected')} 与路径数 {idx.size} 不一致，使用路径数。",
            file=sys.stderr,
        )
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description="在 ranked test 子集上生成带 bootstrap CI 的 comparison 表")
    ap.add_argument("--comparison-root", type=Path, default=Path("checkpoints/old_data_supcon_compare_v2"))
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="默认: <comparison-root>/test_subset_ranked_cas_first_manifest.json",
    )
    ap.add_argument("--test-dir", type=Path, default=None, help="默认从 manifest source_test_dir 读取")
    ap.add_argument(
        "--comparison-csv",
        type=Path,
        default=None,
        help="模型列表来源，默认 <comparison-root>/comparison_summary.csv",
    )
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--legacy-val-resize",
        action="store_true",
        help="验证/测试使用 Resize((img_size,img_size))，复现 old_data_supcon_compare_v2 指标",
    )
    args = ap.parse_args()

    comp_root = args.comparison_root.resolve()
    manifest_path = (args.manifest or (comp_root / "test_subset_ranked_cas_first_manifest.json")).resolve()
    if not manifest_path.is_file():
        print(f"找不到 manifest: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    proj = infer_project_root(comp_root)
    test_root = args.test_dir
    if test_root is None:
        src = manifest.get("source_test_dir")
        if not src:
            print("manifest 无 source_test_dir，请传 --test-dir", file=sys.stderr)
            sys.exit(1)
        test_root = Path(src)
    test_root = test_root.resolve() if test_root.is_absolute() else (proj / test_root).resolve()
    if not test_root.is_dir():
        print(f"找不到测试目录: {test_root}", file=sys.stderr)
        sys.exit(1)

    csv_in = (args.comparison_csv or (comp_root / "comparison_summary.csv")).resolve()
    if not csv_in.is_file():
        print(f"找不到: {csv_in}", file=sys.stderr)
        sys.exit(1)

    base_ds = ImageFolderDataset(str(test_root), transform=None)
    subset_idx = manifest_paths_to_indices(manifest, test_root, base_ds.samples)
    print(f"子集样本数: {len(subset_idx)}  (manifest: {manifest_path.name})")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_map = load_comparison_dir_map(comp_root)
    names = read_models_column(csv_in)

    fieldnames = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    rows: list[dict] = []
    point_rows: list[dict] = []
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
                augmentation="standard",
                img_size=224,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                legacy_val_resize=args.legacy_val_resize,
            )
        except Exception as e:
            failures.append(f"{name}: {e}")
            continue

        yt = yt[subset_idx]
        yhat = yhat[subset_idx]
        probs = probs[subset_idx]

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
        det["per_class_subset"] = pc_rows
        det["n_subset"] = int(len(yt))
        details[name] = det
        rows.append(row)

        macro_pt, _ = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
        point_auc = compute_macro_auc_ovr(yt, probs)
        point_rows.append(
            {
                "model": name,
                "macro_auc_ovr": f"{point_auc:.6f}" if point_auc is not None else "",
                "acc": f"{float(np.mean(yt == yhat)):.6f}",
                "sensitivity": f"{macro_pt['sensitivity']:.6f}",
                "specificity": f"{macro_pt['specificity']:.6f}",
                "npv": f"{macro_pt['npv']:.6f}",
                "ppv": f"{macro_pt['ppv']:.6f}",
            }
        )

    if failures:
        print("\n下列模型未完成评估:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
    if not rows:
        print("无有效结果。", file=sys.stderr)
        sys.exit(1)

    rows.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
    for r in rows:
        r.pop("_auc_sort", None)
    point_rows.sort(key=lambda x: -float(x.get("macro_auc_ovr") or 0))

    out_csv = comp_root / "comparison_summary_test_subset.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写入: {out_csv}")

    point_csv = comp_root / "comparison_summary_test_subset_point.csv"
    point_fields = ["model", "macro_auc_ovr", "sensitivity", "specificity", "npv", "ppv", "acc"]
    with point_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=point_fields)
        w.writeheader()
        w.writerows(point_rows)
    print(f"已写入: {point_csv}")

    detail_path = comp_root / "comparison_summary_test_subset_detail.json"
    detail_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写入: {detail_path}")

    pc_csv = comp_root / "comparison_summary_test_subset_per_class.csv"
    pc_fieldnames = ["experiment"] + PER_CLASS_COMPARISON_FIELDS
    with pc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pc_fieldnames)
        w.writeheader()
        for r in per_class_stacked:
            w.writerow({k: r.get(k, "") for k in pc_fieldnames})
    print(f"已写入: {pc_csv}")

    xlsx_path = comp_root / "comparison_summary_test_subset.xlsx"
    try:
        macro_df = pd.DataFrame(rows)
        pc_df = pd.DataFrame(per_class_stacked, columns=pc_fieldnames) if per_class_stacked else pd.DataFrame(columns=pc_fieldnames)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            macro_df.to_excel(writer, sheet_name="comparison_summary", index=False)
            pc_df.to_excel(writer, sheet_name="per_class", index=False)
        print(f"已写入: {xlsx_path}")
    except ModuleNotFoundError as e:
        if "openpyxl" in str(e):
            print("未安装 openpyxl，跳过 xlsx")
        else:
            raise


if __name__ == "__main__":
    main()
