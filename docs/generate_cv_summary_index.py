#!/usr/bin/env python3
"""扫描仓库内全部 cv_summary.json，生成分类索引（Markdown + CSV）。

用法（在仓库根目录）:
    python3 docs/generate_cv_summary_index.py

输出:
    docs/all_cv_summary_index.md
    docs/all_cv_summary_index.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def category_for(rel: str) -> str:
    s = rel.replace("\\", "/")
    padded = f"/{s}/"
    if s.startswith("output/"):
        return "output"
    if "/new_data_models/" in padded:
        return "new_data"
    if "/ablation_study/" in padded:
        return "ablation"
    if (
        "/clip_models/" in padded
        or "/clip_agent_v2/" in padded
        or "/comparison_experiments/" in padded
    ):
        return "clip"
    if "/final_models/" in padded or "/final_starnet_models/" in padded:
        return "final_*"
    return "other"


def main() -> None:
    paths = sorted(ROOT.rglob("cv_summary.json"))
    buckets: dict[str, list[str]] = {
        "output": [],
        "ablation": [],
        "clip": [],
        "new_data": [],
        "final_*": [],
        "other": [],
    }
    for p in paths:
        rel = p.relative_to(ROOT).as_posix()
        buckets[category_for(rel)].append(rel)

    order = ["output", "ablation", "clip", "new_data", "final_*", "other"]

    md_lines = [
        "# cv_summary.json 全量索引",
        "",
        f"自动生成：共 **{len(paths)}** 个文件（相对仓库根目录路径）。",
        "",
        "重新生成：在仓库根目录执行 `python3 docs/generate_cv_summary_index.py`。",
        "",
        "分类规则（按顺序匹配第一条）：`output/` → `new_data_models` → `ablation_study` → "
        "`clip_models` / `clip_agent_v2` / `comparison_experiments` → "
        "`final_models` / `final_starnet_models` → 其余归入 **other**。",
        "",
        "同级 CSV：`docs/all_cv_summary_index.csv`（列：`category,relative_path`）。",
        "",
    ]

    for name in order:
        items = buckets[name]
        title = name if name != "final_*" else "final_*（final_models / final_starnet_models）"
        md_lines.append(f"## {title}（{len(items)}）")
        md_lines.append("")
        for rel in items:
            md_lines.append(f"- `{rel}`")
        md_lines.append("")

    docs = ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "all_cv_summary_index.md").write_text("\n".join(md_lines), encoding="utf-8")

    csv_path = docs / "all_cv_summary_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category", "relative_path"])
        for name in order:
            for rel in buckets[name]:
                w.writerow([name, rel])

    counts = {k: len(v) for k, v in buckets.items()}
    print(f"Wrote docs/all_cv_summary_index.md + .csv, total={len(paths)}, counts={counts}")


if __name__ == "__main__":
    main()
