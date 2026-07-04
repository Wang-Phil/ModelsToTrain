#!/usr/bin/env python3
"""
对单个 CasGNet（SupCon+CE）checkpoint，在 old_data/val 与 eltra_test 上推断，
绘制 multiclass OvR ROC（各类一条曲线）与混淆矩阵；两类图 **分文件** 输出。
混淆矩阵：默认按行归一化着色，不标注格内数值（可用 --confusion-percent / --confusion-counts 显示）。

用法（在项目根）:
  python plot_casgnet_val_eltra_roc_confusion.py \\
    --checkpoint checkpoints/casgnet_supcon_olddata_resplit_new/best_auc_model.pth \\
    --val-dir old_data/val \\
    --test-dir eltra_test \\
    --out-dir checkpoints/casgnet_supcon_olddata_resplit_new/plots_val_eltra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as curve_auc, confusion_matrix, roc_curve

from compare_models_on_eltra_test import run_one_checkpoint
from refresh_supcon_checkpoint_metrics import infer_project_root
from train_casgnet_contrastive_newdata import compute_macro_auc_ovr


def _ensure_inside_axes(ax: matplotlib.axes.Axes) -> None:
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)


def plot_ovr_roc(
    probs: np.ndarray,
    yt: np.ndarray,
    class_names: list[str],
    title: str,
    macro_auc: float,
    ax: matplotlib.axes.Axes,
) -> None:
    n_classes = len(class_names)
    cmap = plt.get_cmap("tab10")
    for c in range(n_classes):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            ax.scatter([], [], label=f"{class_names[c]} (n/a)")
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        a = curve_auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            lw=1.8,
            color=cmap(c % 10),
            label=f"{class_names[c]} (AUC={a:.3f})",
        )
    ax.set_title(f"macro OvR AUC = {macro_auc:.4f}" if not title else f"{title}\nmacro OvR AUC = {macro_auc:.4f}")
    _ensure_inside_axes(ax)


def plot_confusion(
    yt: np.ndarray,
    yhat: np.ndarray,
    class_names: list[str],
    title: str,
    ax: matplotlib.axes.Axes,
    *,
    show_fraction: bool = True,
    show_cell_text: bool = False,
) -> None:
    n = len(class_names)
    cm = confusion_matrix(yt, yhat, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if not show_cell_text:
        return

    for i in range(n):
        for j in range(n):
            if show_fraction:
                pct = int(round(cm_norm[i, j] * 100))
                txt = f"{pct}%" if pct > 0 else "0%"
            else:
                txt = str(int(cm[i, j]))
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=tc)


def run_dataset(
    *,
    ck_path: Path,
    data_root: Path,
    dataset_label: str,
    device,
    aug: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    out_dir: Path,
    prefix: str,
    legacy_val_resize: bool = False,
    subset_indices: np.ndarray | None = None,
    show_sample_count: bool = False,
    confusion_fraction: bool = True,
    show_dataset_title: bool = False,
    show_confusion_text: bool = False,
) -> dict:
    probs, yt, yhat, _n_cls, class_names = run_one_checkpoint(
        ck_path,
        data_root,
        device=device,
        augmentation=aug,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        legacy_val_resize=legacy_val_resize,
    )
    if subset_indices is not None:
        probs = probs[subset_indices]
        yt = yt[subset_indices]
        yhat = yhat[subset_indices]
    macro = compute_macro_auc_ovr(yt, probs)
    if show_dataset_title:
        title_full = dataset_label if not show_sample_count else f"{dataset_label} (n={len(yt)})"
    else:
        title_full = ""

    fig_r, ax_r = plt.subplots(figsize=(8.5, 8))
    plot_ovr_roc(probs, yt, class_names, title_full, macro, ax_r)
    fig_r.tight_layout()
    outp_r = out_dir / f"{prefix}_roc.png"
    fig_r.savefig(outp_r, dpi=160, bbox_inches="tight")
    plt.close(fig_r)
    print(f"已写入: {outp_r}")

    fig_c, ax_c = plt.subplots(figsize=(9.5, 8))
    cm_title = title_full if not confusion_fraction else ""
    plot_confusion(
        yt,
        yhat,
        class_names,
        cm_title,
        ax_c,
        show_fraction=confusion_fraction,
        show_cell_text=show_confusion_text,
    )
    fig_c.tight_layout()
    outp_c = out_dir / f"{prefix}_confusion.png"
    fig_c.savefig(outp_c, dpi=160, bbox_inches="tight")
    plt.close(fig_c)
    print(f"已写入: {outp_c}")
    from train_casgnet_contrastive_newdata import compute_macro_classification_metrics

    macro_pt, _ = compute_macro_classification_metrics(yt, yhat, n_classes=len(class_names))
    return {
        "dataset": dataset_label,
        "n_samples": int(len(yt)),
        "macro_auc_ovr": float(macro),
        "macro_acc_ovr": float(macro_pt["acc"]),
        "multiclass_accuracy": float(np.mean(yt == yhat)),
        "roc_png": str(outp_r),
        "confusion_png": str(outp_c),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CasGNet: val & eltra_test — ROC 与混淆矩阵分文件输出")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/casgnet_supcon_olddata_resplit_new/best_auc_model.pth"),
    )
    ap.add_argument("--val-dir", type=Path, default=Path("old_data/val"))
    ap.add_argument("--test-dir", type=Path, default=Path("eltra_test"))
    ap.add_argument("--project-root", type=Path, default=None)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="默认: <checkpoint 父目录>/plots_val_eltra",
    )
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--legacy-val-resize",
        action="store_true",
        help="使用 Resize((img_size,img_size))，复现 old_data_supcon_compare_v2 指标",
    )
    ap.add_argument(
        "--skip-val",
        action="store_true",
        help="跳过 old_data/val 绘图",
    )
    ap.add_argument(
        "--skip-test",
        action="store_true",
        help="跳过 test-dir 绘图",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="若指定，则在 test-dir 上仅绘制 manifest 子集（如 test_subset manifest）",
    )
    ap.add_argument(
        "--plot-prefix",
        type=str,
        default=None,
        help="test-dir 全量评估时的输出文件名前缀（默认: test 目录名）",
    )
    ap.add_argument(
        "--manifest-prefix",
        type=str,
        default="test_subset",
        help="使用 --manifest 时的输出文件名前缀（可被 --plot-prefix 覆盖）",
    )
    ap.add_argument(
        "--val-prefix",
        type=str,
        default="old_data_val",
        help="val-dir 评估时的输出文件名前缀",
    )
    ap.add_argument(
        "--show-sample-count",
        action="store_true",
        help="在标题中显示样本量 n=...（默认不显示）",
    )
    ap.add_argument(
        "--confusion-counts",
        action="store_true",
        help="混淆矩阵格内标注样本数",
    )
    ap.add_argument(
        "--confusion-percent",
        action="store_true",
        help="混淆矩阵格内标注行归一化整数百分比",
    )
    ap.add_argument(
        "--show-dataset-title",
        action="store_true",
        help="在 ROC/混淆矩阵标题中显示数据集名称（默认不显示）",
    )
    args = ap.parse_args()

    ck_path = args.checkpoint.resolve()
    if not ck_path.is_file():
        print(f"找不到 checkpoint: {ck_path}", file=sys.stderr)
        sys.exit(1)

    proj = args.project_root.resolve() if args.project_root else Path(__file__).resolve().parent
    val_root = args.val_dir.resolve() if args.val_dir.is_absolute() else (proj / args.val_dir).resolve()
    test_root = args.test_dir.resolve() if args.test_dir.is_absolute() else (proj / args.test_dir).resolve()

    out_dir = args.out_dir.resolve() if args.out_dir else ck_path.parent / "plots_val_eltra"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary: list[dict] = []

    show_confusion_text = args.confusion_percent or args.confusion_counts
    confusion_fraction = not args.confusion_counts

    if not args.skip_val:
        summary.append(
            run_dataset(
                ck_path=ck_path,
                data_root=val_root,
                dataset_label="old_data / val",
                device=device,
                aug=args.augmentation,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=out_dir,
                prefix=args.val_prefix,
                legacy_val_resize=args.legacy_val_resize,
                show_sample_count=args.show_sample_count,
                confusion_fraction=confusion_fraction,
                show_dataset_title=args.show_dataset_title,
                show_confusion_text=show_confusion_text,
            )
        )

    if not args.skip_test:
        subset_idx = None
        prefix = args.plot_prefix or (
            "eltra_test" if test_root.name == "eltra_test" else test_root.name.replace("/", "_")
        )
        if args.manifest is not None:
            import json
            from eval_test_subset_bootstrap import manifest_paths_to_indices
            from train_multiclass import ImageFolderDataset

            manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
            base = ImageFolderDataset(str(test_root), transform=None)
            subset_idx = manifest_paths_to_indices(manifest, test_root, base.samples)
            prefix = args.plot_prefix or args.manifest_prefix

        summary.append(
            run_dataset(
                ck_path=ck_path,
                data_root=test_root,
                dataset_label=test_root.name + (" (subset)" if subset_idx is not None else ""),
                device=device,
                aug=args.augmentation,
                img_size=args.img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                out_dir=out_dir,
                prefix=prefix,
                legacy_val_resize=args.legacy_val_resize,
                subset_indices=subset_idx,
                show_sample_count=args.show_sample_count,
                confusion_fraction=confusion_fraction,
                show_dataset_title=args.show_dataset_title,
                show_confusion_text=show_confusion_text,
            )
        )

    if summary:
        import json

        summary_path = out_dir / "plot_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已写入: {summary_path}")


if __name__ == "__main__":
    main()
