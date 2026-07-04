#!/usr/bin/env python3
"""
将 old_data 调整为：
  - test/  ：原 val/（仅改名移动）
  - train/ ：原 train 中约 80%
  - val/   ：原 train 中约 20%（按类别分层，random_state=42）

用法（在 ModelsTotrain 根目录）：
  python3 scripts/reorganize_old_data_train_val_test.py [--dry-run]
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(class_dir: Path) -> list[Path]:
    out = []
    for p in class_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return sorted(out)


def split_train_val(files: list[Path], val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    if not files:
        return [], []
    if len(files) == 1:
        return files, []
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n_val = round(len(shuffled) * val_ratio)
    n_val = max(1, min(len(shuffled) - 1, n_val))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "old_data",
        help="含 train/、val/ 的数据根目录",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="从原 train 划给新 val 的比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root: Path = args.data_root.resolve()
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.is_dir():
        raise SystemExit(f"缺少 train 目录: {train_dir}")
    if not val_dir.is_dir():
        raise SystemExit(f"缺少 val 目录: {val_dir}")
    if test_dir.exists():
        raise SystemExit(f"已存在 test 目录，请先改名或删除: {test_dir}")

    staging = root / "_split_staging_train_orig"
    rng = random.Random(args.seed)

    print(f"data_root={root}")
    print(f"val_ratio={args.val_ratio} seed={args.seed} dry_run={args.dry_run}")
    print()

    # 1) val -> test
    print("Step 1: val/ -> test/")
    if args.dry_run:
        print(f"  [dry-run] would rename {val_dir} -> {test_dir}")
    else:
        val_dir.rename(test_dir)
        print(f"  done: {test_dir}")

    # 2) train -> staging
    print("Step 2: train/ -> _split_staging_train_orig/")
    if args.dry_run:
        print(f"  [dry-run] would rename {train_dir} -> {staging}")
    else:
        train_dir.rename(staging)
        print(f"  done: {staging}")

    # 3) build new train / val from staging
    print("Step 3: split staging -> train/ + val/")
    new_train = root / "train"
    new_val = root / "val"
    if not args.dry_run:
        new_train.mkdir(parents=True, exist_ok=True)
        new_val.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([p for p in staging.iterdir() if p.is_dir()])
    total_tr, total_va = 0, 0
    for cdir in class_dirs:
        cls = cdir.name
        extras = [p.name for p in cdir.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_SUFFIXES]
        if extras:
            raise SystemExit(f"{cls}/ 下存在非图像文件，请先移走: {extras[:5]}...")
        files = list_images(cdir)
        tr_files, va_files = split_train_val(files, args.val_ratio, rng)
        total_tr += len(tr_files)
        total_va += len(va_files)
        print(f"  {cls}: train={len(tr_files)} val={len(va_files)} (from {len(files)})")
        if args.dry_run:
            continue
        (new_train / cls).mkdir(parents=True, exist_ok=True)
        (new_val / cls).mkdir(parents=True, exist_ok=True)
        for p in tr_files:
            shutil.move(str(p), str(new_train / cls / p.name))
        for p in va_files:
            shutil.move(str(p), str(new_val / cls / p.name))

    if not args.dry_run:
        shutil.rmtree(staging)
        print(f"\nRemoved empty staging: {staging}")

    print(f"\nSummary: new train={total_tr} new val={total_va}")
    if not args.dry_run:
        test_count = sum(len(list_images(d)) for d in test_dir.iterdir() if d.is_dir())
        print(f"test (former val) images: {test_count}")
        print("\nDone. Structure: train/ val/ test/")


if __name__ == "__main__":
    main()
