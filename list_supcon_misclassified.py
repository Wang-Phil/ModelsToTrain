#!/usr/bin/env python3
"""
列出 SupCon+CE 训练得到的 CASGNet（等骨干）在验证集（或训练集）上预测错误的图片路径。

划分方式与 train_casgnet_contrastive_newdata.py 一致：stratify + train_test_split(seed, val_ratio)。

用法示例:
  python list_supcon_misclassified.py \\
    --checkpoint checkpoints/casgnet_supcon_newdata/best_auc_model.pth \\
    --data-dir new_data \\
    --output misclassified_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_casgnet_contrastive_newdata import SupConClassifierNet, TransformSubset
from train_multiclass import ImageFolderDataset, get_data_augmentation


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="best_auc_model.pth 等")
    p.add_argument("--data-dir", type=str, default="new_data")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="val", choices=("val", "train", "all"))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--augmentation", type=str, default="standard", help="仅用验证 transform（与训练脚本 val 一致）")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="JSONL；默认打印到 stdout")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    ck_path = Path(args.checkpoint)
    if not ck_path.is_file():
        print(f"找不到权重: {ck_path}", file=sys.stderr)
        sys.exit(1)
    data_root = Path(args.data_dir)
    if not data_root.is_dir():
        print(f"数据目录不存在: {data_root}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(str(ck_path), map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        checkpoint = torch.load(str(ck_path), map_location=device)

    num_classes = int(checkpoint["num_classes"])
    class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    model_name = checkpoint.get("model")
    if not model_name:
        mv = checkpoint.get("model_variant")
        if mv is not None and str(mv).strip():
            model_name = f"casgnet_{str(mv).strip()}"
        else:
            model_name = "casgnet_s1"

    _, val_aug = get_data_augmentation(augmentation_type=args.augmentation, img_size=args.img_size)
    base = ImageFolderDataset(str(data_root), transform=None)  # type: ignore[arg-type]

    if base.class_to_idx != class_to_idx:
        print(
            "警告: 当前 data-dir 的 class_to_idx 与 checkpoint 中不一致，"
            "请使用训练时相同的数据目录与类别文件夹集合。",
            file=sys.stderr,
        )

    n_samples = len(base)
    lab_list = [lab for _, lab in base.samples]
    tr_idx, va_idx = train_test_split(
        np.arange(n_samples),
        test_size=args.val_ratio,
        stratify=lab_list,
        random_state=args.seed,
        shuffle=True,
    )
    if args.split == "val":
        indices = va_idx
    elif args.split == "train":
        indices = tr_idx
    else:
        indices = np.arange(n_samples)

    ds = TransformSubset(base, indices, transform=val_aug)
    pin = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    model = SupConClassifierNet(
        num_classes,
        model_name=model_name,
        pretrained=args.pretrained,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    errors: list[dict] = []
    total = 0
    wrong = 0

    # 建立 global_idx -> path,label（当前 split 内顺序与 loader 一致）
    ordered = [int(indices[i]) for i in range(len(indices))]
    flat_paths: list[str] = []
    flat_labels: list[int] = []
    for gi in ordered:
        path, y = base.samples[gi]
        flat_paths.append(path)
        flat_labels.append(int(y))

    offset = 0
    with torch.inference_mode():
        for images, y in tqdm(loader, desc="Infer"):
            images = images.to(device, non_blocking=True)
            logits = model.forward_logits(images)
            probs = F.softmax(logits.float(), dim=1)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_np = y.numpy()
            pr = probs.cpu().numpy()
            b = y_np.shape[0]
            for i in range(b):
                total += 1
                yi = int(y_np[i])
                pi = int(pred[i])
                if pi != yi:
                    wrong += 1
                    path = flat_paths[offset + i]
                    errors.append(
                        {
                            "path": path,
                            "true_class": idx_to_class[yi],
                            "pred_class": idx_to_class[pi],
                            "true_idx": yi,
                            "pred_idx": pi,
                            "confidence_pred": float(pr[i, pi]),
                            "confidence_true": float(pr[i, yi]),
                        }
                    )
            offset += b

    summary = {
        "checkpoint": str(ck_path),
        "data_dir": str(data_root),
        "split": args.split,
        "n_evaluated": total,
        "n_wrong": wrong,
        "accuracy": (total - wrong) / max(total, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_lines = [json.dumps(e, ensure_ascii=False) for e in errors]
    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        print(f"已写入 {len(errors)} 条误判记录 -> {args.output}", file=sys.stderr)
    else:
        for line in out_lines:
            print(line)


if __name__ == "__main__":
    main()
