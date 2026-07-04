#!/usr/bin/env python3
"""
对 new_data/（各「类名」子目录，ImageFolder 式）全库去重：同一**文件名**若出现在多个类子目录，只保留一份。

保留规则：在排序后的路径中选第一个（(类目录名, 全路径) 升序），删除其余。

若多份非字节相同，删除前在 stderr 给出 [warn] 行（仍只保留一条，见业务风险）。
可先用 --dry-run 查看。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def list_class_files(new_data: Path) -> list[Path]:
    out: list[Path] = []
    for d in sorted(new_data.iterdir()):
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_file() and not f.name.startswith("."):
                out.append(f)
    return out


def run(new_data: Path, dry_run: bool) -> int:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for p in list_class_files(new_data):
        by_name[p.name].append(p)

    n_would_remove = 0
    n_warned = 0
    for name in sorted(by_name):
        paths = by_name[name]
        if len(paths) < 2:
            continue
        sorted_paths = sorted(paths, key=lambda p: (p.parent.name, str(p)))
        keep = sorted_paths[0]
        hashes = {p: sha256_path(p) for p in paths}
        uniq = set(hashes.values())
        if len(uniq) > 1:
            n_warned += 1
            print(
                f"[warn] 同文件名不同内容 {name!r} — 保留 {keep}，将删其余 (hash 不一致)",
                file=sys.stderr,
            )
            for p in sorted_paths[1:]:
                print(
                    f"  drop {p}  sha={hashes[p][:12]}...",
                    file=sys.stderr,
                )
        to_remove = sorted_paths[1:]
        for p in to_remove:
            n_would_remove += 1
            if dry_run:
                print(f"[DRY] DELETE {p} (keep {keep})")
            else:
                p.unlink()
                print(f"DELETE {p} (keep {keep})")

    n_groups = sum(1 for v in by_name.values() if len(v) > 1)
    print(
        f"--- summary: dup_groups={n_groups} "
        f"removed={n_would_remove} different_content_basename={n_warned} dry_run={dry_run}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="全库 new_data 按文件名去重（多类下同名只留一条）")
    ap.add_argument(
        "--new-data",
        type=Path,
        default=Path(__file__).resolve().parent / "new_data",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path().resolve()
    nd = (args.new_data if args.new_data.is_absolute() else root / args.new_data).resolve()
    if not nd.is_dir():
        print(f"not a directory: {nd}", file=sys.stderr)
        return 1
    return run(nd, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
