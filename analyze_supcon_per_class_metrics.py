#!/usr/bin/env python3
"""
对 SupCon+CE 保存的权重，在验证集/训练集/全量上按「多类 one-vs-rest」计算每个类别的:

  - AUC: 以 y==k 为阳性标签、softmax 第 k 列为分数的 ROC-AUC
  - Sensitivity (Recall): TP / (TP+FN)，来自多类混淆矩阵把 k 视为阳性
  - Specificity: TN / (TN+FP)
  - PPV (Precision): TP / (TP+FP)
  - NPV: TN / (TN+FN)
  - ACC_ovr: (TP+TN) / N，即上述二分类视角下的准确率

划分与 train_casgnet_contrastive_newdata.py / list_supcon_misclassified.py 一致。

示例:
  python analyze_supcon_per_class_metrics.py \\
    --checkpoint checkpoints/casgnet_supcon_newdata/best_auc_model.pth \\
    --data-dir new_data \\
    --output-json checkpoints/casgnet_supcon_newdata/per_class_metrics_val.json

  将当前 split 上错分样本按「真实类__to__预测类」复制到子目录:
  python analyze_supcon_per_class_metrics.py ... \\
    --copy-misclassified-to checkpoints/casgnet_supcon_newdata/val_misclassified_copies
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
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


def _infer_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 probs [N,C], y_true [N], y_pred_argmax [N]。"""
    model.eval()
    probs_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    pred_list: list[np.ndarray] = []
    m = model.module if hasattr(model, "module") else model
    use_ac = use_amp and device.type == "cuda"
    for images, y in tqdm(loader, desc="[Infer]"):
        images = images.to(device, non_blocking=True)
        if use_ac:
            with torch.amp.autocast("cuda"):
                logits = m.forward_logits(images)
        else:
            logits = m.forward_logits(images)
        pr = F.softmax(logits.float(), dim=1).cpu().numpy()
        probs_list.append(pr)
        y_list.append(y.numpy())
        pred_list.append(np.argmax(pr, axis=1))
    return np.vstack(probs_list), np.concatenate(y_list), np.concatenate(pred_list)


def per_class_metrics_ovr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> tuple[list[dict], dict]:
    """
    由多类 argmax 预测构造每个类别 k 的 OvR 混淆表 (TP,FP,FN,TN)，
    并计算 AUC / Sens / Spec / PPV / NPV / ACC_ovr。
    """
    labels = np.arange(len(class_names))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_total = int(cm.sum())
    n_classes = len(class_names)
    rows: list[dict] = []

    for k in range(n_classes):
        tp = int(cm[k, k])
        fn = int(cm[k, :].sum() - tp)
        fp = int(cm[:, k].sum() - tp)
        tn = n_total - tp - fn - fp

        def safe_div(a: float, b: float) -> float | None:
            if b <= 0:
                return None
            return float(a / b)

        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        ppv = safe_div(tp, tp + fp)
        npv = safe_div(tn, tn + fn)
        acc_ovr = safe_div(tp + tn, n_total)

        y_bin = (y_true == k).astype(np.int32)
        auc_k: float | None
        try:
            if len(np.unique(y_bin)) < 2:
                auc_k = None
            else:
                auc_k = float(roc_auc_score(y_bin, y_score[:, k]))
        except (ValueError, TypeError):
            auc_k = None

        rows.append(
            {
                "class": class_names[k],
                "class_idx": k,
                "support_true_k": tp + fn,
                "support_pred_k": tp + fp,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "AUC": auc_k,
                "Sensitivity": sens,
                "Specificity": spec,
                "PPV": ppv,
                "NPV": npv,
                "ACC_ovr": acc_ovr,
            }
        )

    def macro_mean(key: str) -> float | None:
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    summary = {
        "n_samples": n_total,
        "n_classes": n_classes,
        "overall_top1_accuracy": float(np.trace(cm) / max(n_total, 1)),
        "macro_AUC": macro_mean("AUC"),
        "macro_Sensitivity": macro_mean("Sensitivity"),
        "macro_Specificity": macro_mean("Specificity"),
        "macro_PPV": macro_mean("PPV"),
        "macro_NPV": macro_mean("NPV"),
        "macro_ACC_ovr": macro_mean("ACC_ovr"),
    }
    return rows, summary


def _copy_misclassified_images(
    dest_root: Path,
    paths_ordered: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """
    将 y_true != y_pred 的样本复制到 dest_root / '{true}__to__{pred}' / 文件名。
    与 DataLoader(..., shuffle=False) 在 TransformSubset 上的行序一致。
    """
    n = len(y_true)
    if len(paths_ordered) != n:
        raise ValueError(f"path 行数 {len(paths_ordered)} 与标签 {n} 不一致")
    dest_root = dest_root.resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i in range(n):
        if int(y_true[i]) == int(y_pred[i]):
            continue
        tname = class_names[int(y_true[i])]
        pname = class_names[int(y_pred[i])]
        sub = dest_root / f"{tname}__to__{pname}"
        sub.mkdir(parents=True, exist_ok=True)
        src = Path(paths_ordered[i])
        dst = sub / src.name
        if dst.is_file():
            stem, suf = src.stem, src.suffix
            k = 1
            while True:
                cand = sub / f"{stem}_dup{k}{suf}"
                if not cand.is_file():
                    dst = cand
                    break
                k += 1
        shutil.copy2(src, dst)
        copied += 1
    print(f"\n已复制 {copied} 个错分样本到 {dest_root}（按 真实类__to__预测类 分子目录）", file=sys.stderr)


def _print_table(rows: list[dict]) -> None:
    headers = [
        "class",
        "n_true",
        "AUC",
        "Sens",
        "Spec",
        "PPV",
        "NPV",
        "ACC*",
    ]
    col_w = [28, 7, 8, 8, 8, 8, 8, 8]

    def fmt(x: float | None, nd: int = 4) -> str:
        if x is None:
            return "   —   "
        return f"{x:.{nd}f}".rjust(7)

    line = " ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        n_true = r["support_true_k"]
        print(
            f"{str(r['class'])[:26]:<28} "
            f"{str(n_true):>7} "
            f"{fmt(r['AUC']):>8} "
            f"{fmt(r['Sensitivity']):>8} "
            f"{fmt(r['Specificity']):>8} "
            f"{fmt(r['PPV']):>8} "
            f"{fmt(r['NPV']):>8} "
            f"{fmt(r['ACC_ovr']):>8}"
        )
    print("* ACC_ovr = (TP+TN)/N for class k vs not-k (OvR from multiclass CM).")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, default="new_data")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", type=str, default="val", choices=("val", "train", "all"))
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--augmentation", type=str, default="standard")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-json", type=str, default=None)
    p.add_argument(
        "--output-xlsx",
        type=str,
        default=None,
        help="导出 Excel（需 openpyxl）；可与 --output-json 同时使用",
    )
    p.add_argument(
        "--copy-misclassified-to",
        type=str,
        default=None,
        help="若设置：将当前 split 上预测错误的图像复制到此目录下，子目录名为「真实类__to__预测类」",
    )
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
    class_names = [name for name, _idx in sorted(class_to_idx.items(), key=lambda x: x[1])]

    model_name = _resolve_model_name(checkpoint)
    _, val_aug = get_data_augmentation(augmentation_type=args.augmentation, img_size=args.img_size)
    base = ImageFolderDataset(str(data_root), transform=None)  # type: ignore[arg-type]

    if base.class_to_idx != class_to_idx:
        print(
            "警告: data-dir 的 class_to_idx 与 checkpoint 不一致，指标可能无效。",
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

    model = SupConClassifierNet(
        num_classes,
        model_name=model_name,
        pretrained=args.pretrained,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    use_amp = bool(args.use_amp and device.type == "cuda")
    paths_ordered = [str(base.samples[int(indices[i])][0]) for i in range(len(indices))]
    probs, y_true, y_pred = _infer_probs(model, loader, device, use_amp)
    per_rows, summary = per_class_metrics_ovr(y_true, probs, y_pred, class_names)

    print(f"checkpoint: {ck_path}")
    print(f"split: {args.split}  n={summary['n_samples']}  model: {model_name}")
    print(f"overall top-1 ACC: {summary['overall_top1_accuracy']:.4f}")

    def _fmt_m(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    ma = summary["macro_AUC"]
    auc_s = f"{ma:.4f}" if ma is not None else "N/A"
    print(
        "macro (unweighted mean over classes; AUC skips None): "
        f"AUC={auc_s} Sens={_fmt_m(summary['macro_Sensitivity'])} Spec={_fmt_m(summary['macro_Specificity'])} "
        f"PPV={_fmt_m(summary['macro_PPV'])} NPV={_fmt_m(summary['macro_NPV'])} ACC_ovr={_fmt_m(summary['macro_ACC_ovr'])}"
    )

    print()
    _print_table(per_rows)

    out_obj = {
        "checkpoint": str(ck_path),
        "data_dir": str(data_root),
        "split": args.split,
        "model": model_name,
        "summary": summary,
        "per_class": per_rows,
        "notes": {
            "AUC": "OvR binary ROC-AUC for class k vs softmax score[:,k].",
            "Sensitivity_PPV_NPV_Specificity_ACC_ovr": (
                "Derived from multiclass confusion matrix by treating class k as positive: "
                "TP=cm[k,k], FN=row_sum-TP, FP=col_sum-TP, TN=N-TP-FN-FP."
            ),
            "overall_top1_accuracy": "argmax accuracy on multiclass task.",
        },
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)
        print(f"\n已写入 {args.output_json}", file=sys.stderr)

    if args.output_xlsx:
        from export_per_class_metrics_excel import write_per_class_metrics_excel

        write_per_class_metrics_excel(out_obj, args.output_xlsx)
        print(f"已写入 Excel: {args.output_xlsx}", file=sys.stderr)

    if args.copy_misclassified_to:
        _copy_misclassified_images(
            Path(args.copy_misclassified_to),
            paths_ordered,
            y_true,
            y_pred,
            class_names,
        )


if __name__ == "__main__":
    main()
