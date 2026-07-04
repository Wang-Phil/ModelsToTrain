#!/usr/bin/env python3
"""
针对指定类别（默认 Acetabular Loosening 与 Good Place），在验证集上分析:
  - 真为该类的样本被预测成哪一类（混淆去向）
  - 两类之间的相互误判列表（路径 + 概率）
  - 所有「真实或预测」涉及关注类且预测错误的样本

划分与 list_supcon_misclassified.py 一致。

示例:
  python analyze_focus_class_confusion.py \\
    --checkpoint checkpoints/casgnet_supcon_newdata/best_auc_model.pth \\
    --data-dir new_data \\
    --focus Acetabular Loosening Good Place \\
    --output-dir checkpoints/casgnet_supcon_newdata/focus_confusion
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_casgnet_contrastive_newdata import SupConClassifierNet, TransformSubset
from train_multiclass import ImageFolderDataset, get_data_augmentation


def _resolve_model_name(checkpoint: dict) -> str:
    name = checkpoint.get("model")
    if name:
        return str(name)
    mv = checkpoint.get("model_variant")
    if mv is not None and str(mv).strip():
        return f"casgnet_{str(mv).strip()}"
    return "casgnet_s1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default="new_data")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", type=str, default="val", choices=("val", "train", "all"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--focus",
        type=str,
        nargs="+",
        default=["Acetabular Loosening", "Good Place"],
        help="关注的类别名（须与文件夹名一致）",
    )
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck_path = Path(args.checkpoint)
    data_root = Path(args.data_dir)
    if not ck_path.is_file() or not data_root.is_dir():
        print("checkpoint 或 data-dir 无效", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(str(ck_path), map_location=device, weights_only=True)  # type: ignore[call-arg]
    except TypeError:
        checkpoint = torch.load(str(ck_path), map_location=device)

    num_classes = int(checkpoint["num_classes"])
    class_to_idx: dict[str, int] = dict(checkpoint["class_to_idx"])
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    for name in args.focus:
        if name not in class_to_idx:
            print(f"错误: 类别 '{name}' 不在 checkpoint 的 class_to_idx 中。", file=sys.stderr)
            print(f"可用: {list(class_to_idx.keys())}", file=sys.stderr)
            sys.exit(1)

    focus_idx = {class_to_idx[n]: n for n in args.focus}
    focus_set = set(focus_idx.keys())

    model_name = _resolve_model_name(checkpoint)
    _, val_aug = get_data_augmentation(augmentation_type=args.augmentation, img_size=args.img_size)
    base = ImageFolderDataset(str(data_root), transform=None)  # type: ignore[arg-type]

    n_samples = len(base)
    lab_list = [lab for _, lab in base.samples]
    tr_idx, va_idx = train_test_split(
        np.arange(n_samples),
        test_size=args.val_ratio,
        stratify=lab_list,
        random_state=args.seed,
        shuffle=True,
    )
    indices = va_idx if args.split == "val" else tr_idx if args.split == "train" else np.arange(n_samples)

    ds = TransformSubset(base, indices, transform=val_aug)
    pin = device.type == "cuda"
    pkw: dict = {}
    if args.num_workers > 0:
        pkw["persistent_workers"] = True
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        **pkw,
    )

    model = SupConClassifierNet(num_classes, model_name=model_name, pretrained=args.pretrained).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    ordered = [int(indices[i]) for i in range(len(indices))]
    paths_order: list[str] = []
    for gi in ordered:
        paths_order.append(base.samples[gi][0])

    records: list[dict] = []
    offset = 0
    with torch.inference_mode():
        for images, y in tqdm(loader, desc="[Infer]"):
            images = images.to(device, non_blocking=True)
            logits = model.forward_logits(images)
            probs = F.softmax(logits.float(), dim=1).cpu().numpy()
            pred = np.argmax(probs, axis=1)
            y_np = y.numpy().astype(int)
            b = y_np.shape[0]
            for i in range(b):
                yi, pi = int(y_np[i]), int(pred[i])
                pr = probs[i]
                top3 = np.argsort(pr)[::-1][:3].tolist()
                records.append(
                    {
                        "path": paths_order[offset + i],
                        "true_idx": yi,
                        "true_class": idx_to_class[yi],
                        "pred_idx": pi,
                        "pred_class": idx_to_class[pi],
                        "correct": yi == pi,
                        "p_true": float(pr[yi]),
                        "p_pred": float(pr[pi]),
                        "top3_idx": top3,
                        "top3_class": [idx_to_class[j] for j in top3],
                        "top3_prob": [float(pr[j]) for j in top3],
                    }
                )
            offset += b

    # --- 混淆统计：真类为关注类之一时，预测分布（仅错误子集也可看全量） ---
    def breakdown_when_true_is(true_idx: int) -> tuple[Counter, list[dict]]:
        wrong = [r for r in records if r["true_idx"] == true_idx and not r["correct"]]
        c = Counter(r["pred_class"] for r in wrong)
        return c, wrong

    summary: dict = {
        "checkpoint": str(ck_path),
        "data_dir": str(data_root),
        "split": args.split,
        "n_evaluated": len(records),
        "focus_classes": args.focus,
        "when_true_breakdown": {},
        "mutual_confusion_counts": {},
    }

    for ti, tname in focus_idx.items():
        cnt, wrong_list = breakdown_when_true_is(ti)
        summary["when_true_breakdown"][tname] = dict(cnt.most_common())
        (out_dir / f"errors_true_is__{tname.replace(' ', '_')}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in wrong_list) + ("\n" if wrong_list else ""),
            encoding="utf-8",
        )

    # 若恰好两个关注类：相互误判
    if len(args.focus) == 2:
        n0, n1 = args.focus[0], args.focus[1]
        i0, i1 = class_to_idx[n0], class_to_idx[n1]
        to_01 = [r for r in records if r["true_idx"] == i0 and r["pred_idx"] == i1]
        to_10 = [r for r in records if r["true_idx"] == i1 and r["pred_idx"] == i0]
        summary["mutual_confusion_counts"] = {
            f"{n0} -> {n1}": len(to_01),
            f"{n1} -> {n0}": len(to_10),
        }
        safe = lambda s: s.replace(" ", "_").replace("/", "_")
        (out_dir / f"mutual_{safe(n0)}__to__{safe(n1)}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in to_01) + ("\n" if to_01 else ""),
            encoding="utf-8",
        )
        (out_dir / f"mutual_{safe(n1)}__to__{safe(n0)}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in to_10) + ("\n" if to_10 else ""),
            encoding="utf-8",
        )

    # 涉及关注类任一（真或预测）且判错
    involved = [
        r
        for r in records
        if not r["correct"] and (r["true_idx"] in focus_set or r["pred_idx"] in focus_set)
    ]
    (out_dir / "errors_involving_focus_classes.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in involved) + ("\n" if involved else ""),
        encoding="utf-8",
    )

    # 全量混淆矩阵 (counts) 供参考
    cm = defaultdict(lambda: defaultdict(int))
    for r in records:
        cm[r["true_class"]][r["pred_class"]] += 1
    summary["confusion_matrix_true_vs_pred"] = {t: dict(cm[t]) for t in sorted(cm.keys())}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in summary.items() if k != "confusion_matrix_true_vs_pred"}, indent=2, ensure_ascii=False))
    print(f"\n已写入目录: {out_dir.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
