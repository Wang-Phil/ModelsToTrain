#!/usr/bin/env python3
"""
将 val 错分副本与医生标注回写到 new_data/。

仅处理 val_misclassified_copies 里出现的图；new_data 中未参与该次医生复核的样本一律不碰。

对每条错分，子目录名为「{真实类 T}__to__{预测类}」，只操作一条路径 new_data / T / 文件名。

规则:
- 多标签（CSV 中逗号分隔多类）: 删除 new_data / T / 文件（该错分原标注位置）。
- CSV 未出现: 同上，仅删 new_data / T / 文件。
- 单标签: 若 医生标签 D == T，不改动。若 D != T，将 new_data / T / 文件 移到 new_data / D / 文件
  （若目标已存在则先删除再移入，表示以医生纠正为准）。

见 analyze_supcon_per_class_metrics.py 中错分目录命名。

**第二遍 `cleanup_extra_copies`（默认不跑）** — `--cleanup-misclass-basenames`；仅跑第二遍:
`--only-cleanup`。**仅**处理「医生单标签且 **D≠T**」的错分：保留 `D/name`，删除
其它类目录下同名（含仍残留在 `T/` 的副本）。**不**处理 D=T、未标 CSV、多标签。
主流程通常已移好；第二遍用于清跨类重名残留。

"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path


def build_name_to_t(mis_root: Path) -> dict[str, str]:
    tmap: dict[str, str] = {}
    for sub in sorted(mis_root.iterdir()):
        if not sub.is_dir():
            continue
        t, _p = parse_t_pred(sub.name)
        if t is None:
            continue
        for img in sub.iterdir():
            if not img.is_file() or img.suffix.lower() == ".csv":
                continue
            n = img.name
            if n in tmap and tmap[n] != t:
                raise SystemExit(
                    f"错分目录中同一文件名出现不同 T: {n!r} -> {tmap[n]!r} 与 {t!r}"
                )
            tmap[n] = t
    return tmap


def canonical_path(
    name: str,
    t: str,
    ann: dict[str, list[str]],
    new_data: Path,
) -> Path | None:
    """单标签下 canonical；None = 未标或多标。第二遍 cleanup 已不再使用 None 分支。"""
    if name not in ann:
        return None
    labs = ann[name]
    if len(labs) != 1:
        return None
    d = labs[0]
    if d == t:
        return new_data / t / name
    return new_data / d / name


def find_class_subfiles(new_data: Path, name: str) -> list[Path]:
    out: list[Path] = []
    for d in new_data.iterdir():
        if not d.is_dir():
            continue
        p = d / name
        if p.is_file():
            out.append(p)
    return out


def cleanup_extra_copies(
    *,
    new_data: Path,
    name_to_t: dict[str, str],
    ann: dict[str, list[str]],
    dry_run: bool,
) -> tuple[int, list[str]]:
    """
    仅当 单标签 D≠T：保留 D/name，删其它类下同名（含 T）。不读文件、不比较哈希。
    """
    n = 0
    warnings: list[str] = []
    for name, t in name_to_t.items():
        if name not in ann or len(ann[name]) != 1:
            continue
        d = ann[name][0]
        if d == t:
            continue
        can = new_data / d / name
        paths = find_class_subfiles(new_data, name)
        if not paths:
            continue
        if not can.is_file():
            warnings.append(f"[skip cleanup] D≠T 但缺少 {can} (name={name!r})")
            continue
        for p in paths:
            if p.resolve() == can.resolve():
                continue
            n += 1
            if dry_run:
                print(f"[DRY] DELETE (cleanup D≠T) {p} (keep {can})")
            else:
                p.unlink()
                print(f"DELETE (cleanup D≠T) {p} (keep {can})")
    return n, warnings


def parse_t_pred(stem: str) -> tuple[str | None, str | None]:
    if "__to__" not in stem:
        return None, None
    t, p = stem.rsplit("__to__", 1)
    return t.strip(), p.strip()


def load_csv_annotations(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 2:
                continue
            name = row[0].strip()
            raw = row[1].strip().strip('"')
            m = re.match(r"\[([^\]]*)\]\s*$", raw)
            inner = m.group(1) if m else raw.strip("[]")
            labels = [x.strip() for x in inner.split(",") if x.strip()]
            if name:
                out[name] = labels
    return out


def apply_rules(
    *,
    new_data: Path,
    mis_root: Path,
    csv_path: Path,
    dry_run: bool,
    only_cleanup: bool = False,
    do_cleanup: bool = False,
) -> int:
    ann = load_csv_annotations(csv_path)
    n_delete = 0
    n_move = 0
    n_noop = 0
    n_dup = 0
    warnings: list[str] = []

    name_to_t = build_name_to_t(mis_root)

    if not only_cleanup:
        for name, t in sorted(name_to_t.items(), key=lambda x: x[0]):
            src = new_data / t / name
            if name not in ann:
                if not src.is_file():
                    warnings.append(f"[skip] CSV 无此项且路径不存在: {src} (T={t})")
                    continue
                n_delete += 1
                if dry_run:
                    print(f"[DRY] DELETE {src}")
                else:
                    src.unlink()
                    print(f"DELETE {src}")
                continue

            labs = ann[name]
            if len(labs) != 1:
                if not src.is_file():
                    warnings.append(
                        f"[skip] 多标签但路径不存在: {src} labels={labs}"
                    )
                    continue
                n_delete += 1
                if dry_run:
                    print(f"[DRY] DELETE (multi) {src}")
                else:
                    src.unlink()
                    print(f"DELETE (multi) {src}")
                continue

            dlab = labs[0]
            if dlab == t:
                n_noop += 1
                continue

            dest = new_data / dlab / name
            # 幂等: 已在目标、源已无
            if not src.is_file() and dest.is_file():
                continue
            if not src.is_file():
                warnings.append(f"[skip] 需从 T 移出但路径不存在: {src} -> {dest}")
                continue
            n_move += 1
            if dry_run:
                print(
                    f"[DRY] MOVE {src} -> {dest}"
                    + (f" (replace dest)" if dest != src and dest.is_file() else "")
                )
            else:
                if dest == src:
                    n_move -= 1
                    warnings.append(f"[weird] src==dest: {src}")
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.is_file():
                    dest.unlink()
                shutil.move(str(src), str(dest))
                print(f"MOVE {src} -> {dest}")

    dupe_w: list[str] = []
    if do_cleanup:
        n_dup, dupe_w = cleanup_extra_copies(
            new_data=new_data, name_to_t=name_to_t, ann=ann, dry_run=dry_run
        )

    print("---", file=sys.stderr)
    print(
        f"summary: deletes={n_delete} moves={n_move} noop={n_noop} "
        f"dup_removals={n_dup} dry_run={dry_run} only_cleanup={only_cleanup}",
        file=sys.stderr,
    )
    for w in warnings + dupe_w:
        print(w, file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--new-data",
        type=Path,
        default=Path(__file__).resolve().parent / "new_data",
    )
    ap.add_argument(
        "--mis-root",
        type=Path,
        default=Path("checkpoints/casgnet_supcon_newdata/val_misclassified_copies"),
    )
    ap.add_argument(
        "--labels-csv",
        type=Path,
        default=Path(
            "checkpoints/casgnet_supcon_newdata/val_misclassified_copies/"
            "labels_my-project-name_2026-04-27-12-23-45.csv"
        ),
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only-cleanup",
        action="store_true",
        help="只跑第二遍：仅 D≠T 时清跨类同名，不跑主流程",
    )
    ap.add_argument(
        "--cleanup-misclass-basenames",
        action="store_true",
        help="第二遍：仅单标签 D≠T 时保留 D/name、删其它类同名（含 T）；主流程后可选",
    )
    args = ap.parse_args()
    if args.only_cleanup and not args.cleanup_misclass_basenames:
        args.cleanup_misclass_basenames = True
    do_cleanup = args.cleanup_misclass_basenames
    root = Path().resolve()
    return apply_rules(
        new_data=(args.new_data if args.new_data.is_absolute() else root / args.new_data).resolve(),
        mis_root=(args.mis_root if args.mis_root.is_absolute() else root / args.mis_root).resolve(),
        csv_path=(args.labels_csv if args.labels_csv.is_absolute() else root / args.labels_csv).resolve(),
        dry_run=args.dry_run,
        only_cleanup=args.only_cleanup,
        do_cleanup=do_cleanup,
    )


if __name__ == "__main__":
    raise SystemExit(main())
