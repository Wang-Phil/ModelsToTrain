#!/usr/bin/env python3
"""根据 checkpoints/.../comparison_summary.csv 与各实验目录生成双表 comparison_summary.xlsx。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from refresh_supcon_checkpoint_metrics import (
    infer_project_root,
    refresh_comparison_all_per_class,
    write_comparison_workbook,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="写入 comparison_summary.xlsx（宏观表 + per_class_val）")
    ap.add_argument(
        "comparison_root",
        type=Path,
        help="含 comparison_summary.csv 的目录，例如 checkpoints/old_data_supcon_compare",
    )
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="含 old_data/val 的项目根（默认从 comparison_root 推断 <repo>/checkpoints/<tag> → <repo>）",
    )
    ap.add_argument(
        "--refresh-all",
        action="store_true",
        help="先对每个模型 reload 权重并做 per-class bootstrap（默认 n=1000，见 --n-bootstrap），再写 xlsx",
    )
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--allow-legacy",
        action="store_true",
        help="仅生成 xlsx 时允许旧版 per_class JSON 回退（无 bootstrap）；不与 --refresh-all 共用",
    )
    args = ap.parse_args()
    root = args.comparison_root.resolve()
    proj = args.project_root.resolve() if args.project_root else infer_project_root(root)

    if args.refresh_all:
        device = torch.device(args.device) if args.device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        refresh_comparison_all_per_class(
            root,
            project_root=proj,
            n_bootstrap=args.n_bootstrap,
            device=device,
            write_xlsx=True,
            allow_legacy_fallback=False,
        )
    else:
        write_comparison_workbook(
            root,
            project_root=proj,
            allow_legacy_fallback=args.allow_legacy,
        )


if __name__ == "__main__":
    main()
