#!/usr/bin/env python3
"""
在任意 ImageFolder 结构目录上评估 best_auc_model.pth（与 train_casgnet_contrastive_newdata 保存格式一致）。

输出:
  - macro-OVR：auc、sensitivity、specificity、npv、ppv、acc（与训练脚本宏平均 OvR 一致）
  - 多分类整体准确率 accuracy
  - macro_metrics.csv、per_class_metrics.csv、summary.json
  - 误判样本 misclassified.csv / .jsonl

用法:
  python eval_checkpoint_on_folder.py \\
    --checkpoint checkpoints/casgnet_supcon_olddata_resplit_new/best_auc_model.pth \\
    --test-dir eltra_test \\
    --output-dir checkpoints/casgnet_supcon_olddata_resplit_new/eltra_test_eval
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from train_casgnet_contrastive_newdata import (
    SupConClassifierNet,
    TransformSubset,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="在独立测试文件夹上评估 SupCon+CE 分类 checkpoint")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--test-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None, help="默认: <checkpoint_parent>/extra_eval_<test-dir名>")
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    ck_path = args.checkpoint.resolve()
    test_root = args.test_dir.resolve()
    if not ck_path.is_file():
        print(f"找不到权重: {ck_path}", file=sys.stderr)
        sys.exit(1)
    if not test_root.is_dir():
        print(f"测试目录不存在: {test_root}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir.resolve() if args.output_dir else ck_path.parent / f"extra_eval_{test_root.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(ck_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ck_path, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt)
    num_classes = int(ckpt.get("num_classes"))
    ck_class_to_idx: dict[str, int] = ckpt.get("class_to_idx") or {}
    model_name = str(ckpt.get("model", "casgnet_s1"))
    proj_dim, hidden_dim = _infer_proj_dims(state_dict)

    _, val_aug = get_data_augmentation(augmentation_type=args.augmentation, img_size=args.img_size)
    base = ImageFolderDataset(str(test_root), transform=None)

    if ck_class_to_idx and base.class_to_idx != ck_class_to_idx:
        print(
            "错误: 测试集文件夹类别与 checkpoint 中 class_to_idx 不一致。\n"
            f"  ckpt:   {ck_class_to_idx}\n"
            f"  test:   {base.class_to_idx}",
            file=sys.stderr,
        )
        sys.exit(1)

    idx_to_class = {int(i): n for n, i in base.class_to_idx.items()}
    n = len(base)
    subset = TransformSubset(base, np.arange(n), transform=val_aug)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

    acc_multiclass = float(np.mean(yt == yhat))
    try:
        auc_macro = compute_macro_auc_ovr(yt, probs)
    except Exception:
        auc_macro = float("nan")

    macro_cls, per_class_rows = compute_macro_classification_metrics(yt, yhat, n_classes=num_classes)

    labels_sorted = [idx_to_class[i] for i in range(num_classes)]
    report = classification_report(yt, yhat, labels=list(range(num_classes)), target_names=labels_sorted, digits=4)
    cm = confusion_matrix(yt, yhat, labels=list(range(num_classes))).tolist()

    mis_rows: list[dict] = []
    for i in range(n):
        yi = int(yt[i])
        pi = int(yhat[i])
        if yi != pi:
            path, _ = base.samples[i]
            mis_rows.append(
                {
                    "image_path": path,
                    "true_class": idx_to_class[yi],
                    "pred_class": idx_to_class[pi],
                    "confidence_pred": float(probs[i, pi]),
                    "confidence_true": float(probs[i, yi]),
                }
            )

    summary = {
        "checkpoint": str(ck_path),
        "test_dir": str(test_root),
        "n_samples": n,
        "n_correct": int(n - len(mis_rows)),
        "n_wrong": len(mis_rows),
        "accuracy": acc_multiclass,
        "auc": auc_macro,
        "macro_auc_ovr": auc_macro,
        "sensitivity": macro_cls["sensitivity"],
        "specificity": macro_cls["specificity"],
        "npv": macro_cls["npv"],
        "ppv": macro_cls["ppv"],
        "acc": macro_cls["acc"],
        "macro_ovr_note": "auc 为 macro-OVR；sensitivity/specificity/npv/ppv/acc 为各类别 one-vs-rest 后宏平均（与 result_summary.json 一致）；accuracy 为多分类整体准确率",
        "confusion_matrix": cm,
        "class_names_row_col": labels_sorted,
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    macro_csv = out_dir / "macro_metrics.csv"
    with macro_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "auc_macro_ovr",
                "sensitivity_macro_ovr",
                "specificity_macro_ovr",
                "npv_macro_ovr",
                "ppv_macro_ovr",
                "acc_macro_ovr",
                "accuracy_multiclass",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "auc_macro_ovr": auc_macro,
                "sensitivity_macro_ovr": macro_cls["sensitivity"],
                "specificity_macro_ovr": macro_cls["specificity"],
                "npv_macro_ovr": macro_cls["npv"],
                "ppv_macro_ovr": macro_cls["ppv"],
                "acc_macro_ovr": macro_cls["acc"],
                "accuracy_multiclass": acc_multiclass,
            }
        )

    per_class_csv = out_dir / "per_class_metrics.csv"
    pc_fields = [
        "class_name",
        "tp",
        "tn",
        "fp",
        "fn",
        "sensitivity",
        "specificity",
        "npv",
        "ppv",
        "acc",
    ]
    with per_class_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=pc_fields)
        w.writeheader()
        for row in per_class_rows:
            ci = int(row["class_idx"])
            w.writerow(
                {
                    "class_name": idx_to_class[ci],
                    **{k: row[k] for k in pc_fields if k != "class_name"},
                }
            )

    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    csv_path = out_dir / "misclassified.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["image_path", "true_class", "pred_class", "confidence_pred", "confidence_true"],
        )
        w.writeheader()
        w.writerows(mis_rows)

    jsonl_path = out_dir / "misclassified.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in mis_rows) + ("\n" if mis_rows else ""), encoding="utf-8")

    print(
        json.dumps(
            {
                "n_samples": summary["n_samples"],
                "n_correct": summary["n_correct"],
                "n_wrong": summary["n_wrong"],
                "accuracy_multiclass": summary["accuracy"],
                "auc_macro_ovr": summary["macro_auc_ovr"],
                "sensitivity_macro_ovr": summary["sensitivity"],
                "specificity_macro_ovr": summary["specificity"],
                "npv_macro_ovr": summary["npv"],
                "ppv_macro_ovr": summary["ppv"],
                "acc_macro_ovr": summary["acc"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"详细结果目录: {out_dir}")
    print(f"  summary.json / macro_metrics.csv / per_class_metrics.csv / classification_report.txt / misclassified.csv")


if __name__ == "__main__":
    main()
