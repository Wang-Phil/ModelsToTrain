from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torchvision import datasets, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate k-fold metrics into mean±std (+95% CI).")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/wangweicheng/ModelsToTrains"),
        help="Project root containing folds/checkpoints.",
    )
    parser.add_argument(
        "--fold-glob",
        type=str,
        default="data_kfold_paper/fold_*",
        help="Glob under root to find fold directories (each must contain test/).",
    )
    parser.add_argument(
        "--ckpt-pattern",
        type=str,
        default="checkpoints/ConvNext_tiny_dual_attention_224*448_fold{fold}/best*.pth",
        help="Checkpoint pattern with '{fold}' placeholder; picks last sorted match.",
    )
    parser.add_argument(
        "--ckpt-fallback",
        type=str,
        default="checkpoints/ConvNext_tiny_dual_attention_224*448/best*.pth",
        help="Fallback checkpoint glob when fold-specific pattern not found.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device string, e.g. cuda:0 or cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="Eval batch size.")
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save aggregated metrics as JSON.",
    )
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    from models.convnextv2_change import convnextv2_tiny

    model = convnextv2_tiny(pretrained=False)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def mean_std_ci(values: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, std, ci95


def compute_weighted_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    supports = cm.sum(axis=1)
    specs = []
    for c in range(num_classes):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP
        FN = cm[c, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        spec_c = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specs.append(spec_c)
    weights = supports / max(1, supports.sum())
    return float(np.sum(weights * np.array(specs)))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    root = args.root.resolve()

    fold_dirs = sorted(glob.glob(str(root / args.fold_glob)))
    if not fold_dirs:
        raise SystemExit(f"No fold directories found with: {root / args.fold_glob}")

    tf = transforms.Compose(
        [
            transforms.Resize((224, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.3785765, 0.37864596, 0.37860626],
                std=[0.26698703, 0.2670074, 0.2669025],
            ),
        ]
    )

    accs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    f1s: List[float] = []
    specs: List[float] = []
    times: List[float] = []
    per_fold: List[dict] = []

    for fold_dir in fold_dirs:
        fold_name = Path(fold_dir).name
        fold_id = "".join(ch for ch in fold_name if ch.isdigit()) or fold_name

        test_dir = Path(fold_dir) / "test"
        if not test_dir.exists():
            print(f"[WARN] Missing test set in {fold_dir}, skipping.")
            continue

        ds = datasets.ImageFolder(str(test_dir), transform=tf)
        num_classes = len(ds.classes)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True
        )

        ckpt_glob = str(root / args.ckpt_pattern.format(fold=fold_id))
        ckpts = sorted(glob.glob(ckpt_glob))
        if not ckpts:
            ckpts = sorted(glob.glob(str(root / args.ckpt_fallback)))
        if not ckpts:
            raise SystemExit(f"No checkpoint found for fold '{fold_id}' with patterns:\n- {ckpt_glob}\n- {root / args.ckpt_fallback}")
        ckpt_path = ckpts[-1]

        model = build_model(num_classes).to(device).eval()
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

        y_true: List[int] = []
        y_pred: List[int] = []
        t0 = time.time()
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                logits = model(x)
                p = logits.argmax(1).cpu().numpy().tolist()
                y_pred += p
                y_true += y.numpy().tolist()
        avg_time = (time.time() - t0) / max(1, len(ds))

        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        acc = 100.0 * accuracy_score(y_true_np, y_pred_np)
        pre = 100.0 * precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        rec = 100.0 * recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        f1 = 100.0 * f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)
        spc = 100.0 * compute_weighted_specificity(y_true_np, y_pred_np, num_classes)

        accs.append(acc)
        precs.append(pre)
        recs.append(rec)
        f1s.append(f1)
        specs.append(spc)
        times.append(avg_time)

        per_fold.append(
            {
                "fold": fold_id,
                "checkpoint": ckpt_path,
                "acc": acc,
                "precision_weighted": pre,
                "recall_weighted": rec,
                "f1_weighted": f1,
                "specificity_weighted": spc,
                "avg_inference_time_sec": avg_time,
            }
        )
        print(f"[{fold_name}] Acc={acc:.2f}  Prec={pre:.2f}  Rec={rec:.2f}  F1={f1:.2f}  Spc={spc:.2f}  t={avg_time:.3f}s")

    m_acc, s_acc, ci_acc = mean_std_ci(accs)
    m_pre, s_pre, ci_pre = mean_std_ci(precs)
    m_rec, s_rec, ci_rec = mean_std_ci(recs)
    m_f1, s_f1, ci_f1 = mean_std_ci(f1s)
    m_spc, s_spc, ci_spc = mean_std_ci(specs)
    avg_t = float(np.mean(times)) if times else 0.0

    print("\n===== Summary (mean ± std) with 95% CI =====")
    print(f"Overall Accuracy     : {m_acc:.2f} ± {s_acc:.2f}   (95% CI: {m_acc-ci_acc:.2f} ~ {m_acc+ci_acc:.2f})")
    print(f"Weighted Precision   : {m_pre:.2f} ± {s_pre:.2f}   (95% CI: {m_pre-ci_pre:.2f} ~ {m_pre+ci_pre:.2f})")
    print(f"Sensitivity (Recall) : {m_rec:.2f} ± {s_rec:.2f}   (95% CI: {m_rec-ci_rec:.2f} ~ {m_rec+ci_rec:.2f})")
    print(f"F1-score             : {m_f1:.2f} ± {s_f1:.2f}   (95% CI: {m_f1-ci_f1:.2f} ~ {m_f1+ci_f1:.2f})")
    print(f"Specificity          : {m_spc:.2f} ± {s_spc:.2f}   (95% CI: {m_spc-ci_spc:.2f} ~ {m_spc+ci_spc:.2f})")
    print(f"Inference time (sec) : {avg_t:.3f}")

    if args.save_json:
        out = {
            "per_fold": per_fold,
            "summary": {
                "acc_mean": m_acc,
                "acc_std": s_acc,
                "acc_ci95": ci_acc,
                "precision_mean": m_pre,
                "precision_std": s_pre,
                "precision_ci95": ci_pre,
                "recall_mean": m_rec,
                "recall_std": s_rec,
                "recall_ci95": ci_rec,
                "f1_mean": m_f1,
                "f1_std": s_f1,
                "f1_ci95": ci_f1,
                "specificity_mean": m_spc,
                "specificity_std": s_spc,
                "specificity_ci95": ci_spc,
                "avg_inference_time_sec": avg_t,
            },
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved summary to {args.save_json}")


if __name__ == "__main__":
    main()


