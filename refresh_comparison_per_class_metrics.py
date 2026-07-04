#!/usr/bin/env python3
"""
对 comparison_summary.csv 中的每个实验目录执行：
reload best_auc_model.pth → 验证集推断 → per-class bootstrap（默认 n=1000）→ 写 result_summary / per_class_metrics_val.*
全部成功后生成双表 comparison_summary.xlsx（第二张含 experiment + 各类 bootstrap CI）。

用法（在项目根目录执行，保证 old_data/train、old_data/val 可解析）:
  python refresh_comparison_per_class_metrics.py checkpoints/old_data_supcon_compare
  python refresh_comparison_per_class_metrics.py checkpoints/old_data_supcon_compare --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from refresh_supcon_checkpoint_metrics import infer_project_root, refresh_comparison_all_per_class


def main() -> None:
    ap = argparse.ArgumentParser(description="对比目录内全部模型 per-class bootstrap + 合并 xlsx")
    ap.add_argument(
        "comparison_root",
        type=Path,
        help="含 comparison_summary.csv 的目录，例如 checkpoints/old_data_supcon_compare",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="解析 train_dir/val_dir 相对路径（默认：<repo>，由 comparison_root 推断）",
    )
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--no-xlsx",
        action="store_true",
        help="只刷新各目录 metrics，不写 comparison_summary.xlsx",
    )
    args = ap.parse_args()

    root = args.comparison_root.resolve()
    proj = args.project_root.resolve() if args.project_root else infer_project_root(root)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    refresh_comparison_all_per_class(
        root,
        project_root=proj,
        augmentation=args.augmentation,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        device=device,
        write_xlsx=not args.no_xlsx,
        allow_legacy_fallback=False,
    )


if __name__ == "__main__":
    main()
