#!/usr/bin/env python3
"""Build FINAL_RELEASE self-contained publication package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "FINAL_RELEASE"
EA = ROOT / "evaluation_results" / "excel_aligned"

MAIN_MODELS = {
    "casgnet": "casgnet_s1_ce_only",
    "starnet": "starnet_s1_ce_only",
    "densenet121": "densenet121_ce_only",
    "resnet18": "resnet18_ce_only",
    "resnet50": "resnet50_ce_only",
    "mobilenetv4_m": "mobilenetv4_m_ce_only",
    "googlenet": "googlenet_ce_only",
    "lsnet_b": "lsnet_b_ce_only",
}

ABLATION_VARIANTS = {
    "starnet_baseline_ab000": ("ab000", {"SA": False, "GRN": False, "SK-UNIT": False}),
    "casgnet_only_skunit_ab001": ("ab001", {"SA": False, "GRN": False, "SK-UNIT": True}),
    "casgnet_only_grn_ab010": ("ab010", {"SA": False, "GRN": True, "SK-UNIT": False}),
    "casgnet_no_sa_ab011": ("ab011", {"SA": False, "GRN": True, "SK-UNIT": True}),
    "casgnet_only_sa_ab100": ("ab100", {"SA": True, "GRN": False, "SK-UNIT": False}),
    "casgnet_no_grn_ab101": ("ab101", {"SA": True, "GRN": False, "SK-UNIT": True}),
    "casgnet_no_skunit_ab110": ("ab110", {"SA": True, "GRN": True, "SK-UNIT": False}),
    "casgnet_full_ab111": ("ab111", {"SA": True, "GRN": True, "SK-UNIT": True}),
}

T1_AUC = {
    "casgnet": 0.962,
    "starnet": 0.952,
    "densenet121": 0.957,
    "resnet18": 0.951,
    "resnet50": 0.917,
    "mobilenetv4_m": 0.918,
    "googlenet": 0.903,
    "lsnet_b": 0.953,
}

T2_AUC = {
    "casgnet": 0.944,
    "starnet": 0.935,
    "densenet121": 0.933,
    "resnet18": 0.934,
    "resnet50": 0.922,
    "mobilenetv4_m": 0.936,
    "googlenet": 0.929,
    "lsnet_b": 0.922,
}

ABLATION_AUC = {
    "starnet_baseline_ab000": 0.952,
    "casgnet_only_skunit_ab001": 0.960,
    "casgnet_only_grn_ab010": 0.950,
    "casgnet_no_sa_ab011": 0.954,
    "casgnet_only_sa_ab100": 0.941,
    "casgnet_no_grn_ab101": 0.951,
    "casgnet_no_skunit_ab110": 0.953,
    "casgnet_full_ab111": 0.962,
}

CODE_MODEL_FILES = [
    "models/casgnet.py",
    "models/starnet.py",
    "models/starnetsk.py",
    "models/classic_models.py",
    "models/mobilenetv4.py",
    "models/multi_head_classifiers.py",
    "models/lsnet_vendor/__init__.py",
    "models/lsnet_vendor/lsnet.py",
    "models/lsnet_vendor/ska.py",
]

CODE_ROOT_FILES = [
    "train_multiclass.py",
    "requirements.txt",
]

CODE_TRAIN_FILES = [
    "train_casgnet_contrastive_newdata.py",
]

CODE_EVAL_FILES = [
    "evaluation_results/excel_aligned/rebuild_original_split.py",
    "evaluation_results/excel_aligned/run_delong_tests.py",
    "evaluation_results/excel_aligned/run_all_models_eval.py",
    "evaluation_results/excel_aligned/generate_plots.py",
    "evaluation_results/excel_aligned/rebuild_ablation_original_split.py",
    "evaluation_results/excel_aligned/metric_ranking_utils.py",
    "evaluation_results/excel_aligned/MODEL_CHECKPOINT_MAPPING.json",
    "eval_checkpoint_on_folder.py",
    "compare_models_on_eltra_test.py",
    "refresh_supcon_checkpoint_metrics.py",
]


def cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def cp_r(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def file_size(p: Path) -> int:
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build_checkpoints(manifest: list) -> None:
    v2_root = ROOT / "checkpoints" / "old_data_supcon_compare_v2"
    v3_root = ROOT / "checkpoints" / "old_data_supcon_compare_v3"
    ab_root = ROOT / "checkpoints" / "starnetsk_sk_kernel_ablation"

    for friendly, ckpt_name in MAIN_MODELS.items():
        out_dir = OUT / "checkpoints" / "main_8_models" / friendly
        # Table1 (test) uses v2; Table2 (val) uses v3
        for split_tag, ckpt_root in [("table1_v2", v2_root), ("table2_v3", v3_root)]:
            src_dir = ckpt_root / ckpt_name
            split_out = out_dir / split_tag
            split_out.mkdir(parents=True, exist_ok=True)
            for fname in ("best_auc_model.pth", "history.json"):
                src = src_dir / fname
                if src.exists():
                    cp(src, split_out / fname)
                    manifest.append({
                        "path": str(split_out / fname),
                        "source": str(src),
                        "size_bytes": src.stat().st_size,
                    })
                else:
                    print(f"MISSING: {src}")

    for friendly, (ab_code, modules) in ABLATION_VARIANTS.items():
        ckpt_name = f"casgnet_s1_{ab_code}_ce_only"
        src_dir = ab_root / ckpt_name
        out_dir = OUT / "checkpoints" / "ablation_8_variants" / friendly
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("best_auc_model.pth", "history.json"):
            src = src_dir / fname
            if src.exists():
                cp(src, out_dir / fname)
                manifest.append({
                    "path": str(out_dir / fname),
                    "source": str(src),
                    "size_bytes": src.stat().st_size,
                })
            else:
                print(f"MISSING: {src}")


def build_data(manifest: list) -> None:
    snap = EA / "original_split_snapshot"
    splits_out = OUT / "data" / "splits"
    splits_out.mkdir(parents=True, exist_ok=True)
    for fname in (
        "test_image_list.txt",
        "val_image_list.txt",
        "test_manifest.json",
        "val_manifest.json",
        "original_split_summary.json",
        "before_after_vs_searched.csv",
    ):
        src = snap / fname
        if src.exists():
            cp(src, splits_out / fname)
            manifest.append({"path": str(splits_out / fname), "source": str(src), "size_bytes": src.stat().st_size})

    # Class names from test dir structure
    class_names = sorted(d.name for d in (ROOT / "old_data" / "test").iterdir() if d.is_dir())
    cn_path = splits_out / "class_names.json"
    cn_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    # Prediction caches
    cache_out = OUT / "data" / "caches"
    cache_out.mkdir(parents=True, exist_ok=True)
    for src_dir, pattern in [
        (EA / "table1_final_package_original" / "caches", "*_test_predictions.npz"),
        (EA / "table2_final_package_original" / "caches", "*_val_predictions.npz"),
    ]:
        for src in sorted(src_dir.glob(pattern)):
            cp(src, cache_out / src.name)
            manifest.append({"path": str(cache_out / src.name), "source": str(src), "size_bytes": src.stat().st_size})

    relaxed = EA / "relaxed_group_counts.json"
    if relaxed.exists():
        cp(relaxed, OUT / "data" / "relaxed_group_counts.json")
        manifest.append({"path": str(OUT / "data" / "relaxed_group_counts.json"), "source": str(relaxed), "size_bytes": relaxed.stat().st_size})

    # Copy test/val images (~92MB total, under 500MB threshold)
    for split in ("test", "val"):
        src = ROOT / "old_data" / split
        dst = OUT / "data" / "images" / split
        if src.exists():
            cp_r(src, dst)
            sz = file_size(dst)
            manifest.append({"path": str(dst), "source": str(src), "size_bytes": sz, "note": "full image split copy"})


def build_code(manifest: list) -> None:
    init_dst = OUT / "code" / "models" / "__init__.py"
    init_dst.parent.mkdir(parents=True, exist_ok=True)
    init_dst.write_text("# Models package for FINAL_RELEASE inference/evaluation pipeline.\n", encoding="utf-8")

    for rel in CODE_MODEL_FILES:
        src = ROOT / rel
        dst = OUT / "code" / rel
        if src.exists():
            cp(src, dst)
            manifest.append({"path": str(dst), "source": str(src), "size_bytes": src.stat().st_size})
        else:
            print(f"MISSING code: {src}")

    for rel in CODE_ROOT_FILES:
        src = ROOT / rel
        dst = OUT / "code" / rel
        if src.exists():
            cp(src, dst)
            manifest.append({"path": str(dst), "source": str(src), "size_bytes": src.stat().st_size})

    for rel in CODE_TRAIN_FILES:
        src = ROOT / rel
        dst = OUT / "code" / "train" / Path(rel).name
        if src.exists():
            cp(src, dst)
            manifest.append({"path": str(dst), "source": str(src), "size_bytes": src.stat().st_size})

    for rel in CODE_EVAL_FILES:
        src = ROOT / rel
        dst = OUT / "code" / "eval" / Path(rel).name
        if src.exists():
            cp(src, dst)
            manifest.append({"path": str(dst), "source": str(src), "size_bytes": src.stat().st_size})
        else:
            print(f"MISSING eval: {src}")


def build_checkpoint_index() -> dict:
    idx = {"main_8_models": {}, "ablation_8_variants": {}}
    for friendly, ckpt_name in MAIN_MODELS.items():
        idx["main_8_models"][friendly] = {
            "checkpoint_name": ckpt_name,
            "table1_ckpt_dir": f"checkpoints/main_8_models/{friendly}/table1_v2",
            "table2_ckpt_dir": f"checkpoints/main_8_models/{friendly}/table2_v3",
            "source_table1": f"checkpoints/old_data_supcon_compare_v2/{ckpt_name}",
            "source_table2": f"checkpoints/old_data_supcon_compare_v3/{ckpt_name}",
            "files": ["best_auc_model.pth", "history.json"],
            "table1_auc": T1_AUC[friendly],
            "table2_auc": T2_AUC[friendly],
        }
    for friendly, (ab_code, modules) in ABLATION_VARIANTS.items():
        idx["ablation_8_variants"][friendly] = {
            "ab_code": ab_code,
            "modules": modules,
            "checkpoint_name": f"casgnet_s1_{ab_code}_ce_only",
            "source": f"checkpoints/starnetsk_sk_kernel_ablation/casgnet_s1_{ab_code}_ce_only",
            "files": ["best_auc_model.pth", "history.json"],
            "table1_auc_original_split": ABLATION_AUC[friendly],
        }
    return idx


def build_data_readme() -> None:
    text = """# 数据说明

## 划分文件 (`splits/`)

| 文件 | 说明 |
|------|------|
| `test_image_list.txt` | 原始 test 集 258 张图像相对路径 |
| `val_image_list.txt` | 原始 val 集 207 张图像相对路径 |
| `test_manifest.json` / `val_manifest.json` | 含类别计数的 manifest |
| `original_split_summary.json` | Version B 汇总（AUC 排名等） |
| `class_names.json` | 7 类名称列表 |

## 图像 (`images/`)

本发布包已包含 `images/test/` 与 `images/val/` 的完整拷贝（约 92 MB）。
若仅需路径列表，也可使用 `splits/*_image_list.txt` 映射回原项目 `old_data/` 目录。

## 预测缓存 (`caches/`)

- `*_test_predictions.npz` — Table1 (n=258) 8 主模型预测
- `*_val_predictions.npz` — Table2 (n=207) 8 主模型预测

可用于快速验证指标而无需重新推理。

## 未包含

- 训练集 `old_data/train/`（体积大，非最终结果必需）
- subset217 搜索子集清单（Version A 补充材料，见 `results/version_A_searched/`）
"""
    (OUT / "data" / "README.md").write_text(text, encoding="utf-8")


def build_master_readme(stats: dict) -> None:
    text = f"""# FINAL_RELEASE — 论文最终结果完整发布包

**生成日期**: {datetime.now().strftime("%Y-%m-%d")}  
**项目根**: `ModelsTotrain/`  
**本目录**: 自包含发布包，所有文件均为真实拷贝（无符号链接）

---

## 1. 文件夹用途

本包汇集髋关节植入物 X 光 7 分类论文的**最终发表结果**所需全部材料：
- 主表结果（Version B，原始 test/val 划分）
- 补充结果（Version A，subset217 搜索对齐）
- 16 个模型 checkpoint（8 主模型 + 8 消融变体）
- 数据划分、预测缓存、评估脚本

## 2. 推荐使用 Version B

**主表请使用 `results/version_B_original/`**（原始 test n=258 / val n=207，无子集搜索）。

Version A（`results/version_A_searched/`）仅作补充/透明性参考，**不应用于主结论**。  
详见 `results/REVIEWER_ASSESSMENT.md`。

## 3. 目录结构

```
FINAL_RELEASE/
├── README.md                 # 本文件
├── MANIFEST.json             # 文件清单（含源路径与大小）
├── results/                  # PUBLICATION_PACKAGE 完整副本
│   ├── version_B_original/   # ★ 主表（Table1/2、消融、DeLong、训练曲线）
│   ├── version_A_searched/   # 补充（subset217）
│   ├── PUBLICATION_MATERIALS_INDEX.md
│   └── REVIEWER_ASSESSMENT.md
├── checkpoints/
│   ├── main_8_models/        # 8 主模型（含 table1_v2 + table2_v3）
│   ├── ablation_8_variants/  # 8 消融变体
│   └── CHECKPOINT_INDEX.json
├── data/
│   ├── splits/               # test/val 划分清单
│   ├── caches/               # 预测缓存
│   ├── images/               # test+val 图像（已拷贝）
│   └── README.md
└── code/
    ├── models/               # 模型定义
    ├── train/                # 训练脚本
    ├── eval/                 # 评估与 DeLong 脚本
    └── requirements.txt
```

## 4. Checkpoint 清单

### 主模型 8 个（Version B AUC）

| 模型 | T1 AUC | T2 AUC | 目录 |
|------|--------|--------|------|
| casgnet | 0.962 | 0.944 | `checkpoints/main_8_models/casgnet/` |
| densenet121 | 0.957 | 0.933 | `checkpoints/main_8_models/densenet121/` |
| lsnet_b | 0.953 | 0.922 | `checkpoints/main_8_models/lsnet_b/` |
| starnet | 0.952 | 0.935 | `checkpoints/main_8_models/starnet/` |
| resnet18 | 0.951 | 0.934 | `checkpoints/main_8_models/resnet18/` |
| mobilenetv4_m | 0.918 | 0.936 | `checkpoints/main_8_models/mobilenetv4_m/` |
| resnet50 | 0.917 | 0.922 | `checkpoints/main_8_models/resnet50/` |
| googlenet | 0.903 | 0.929 | `checkpoints/main_8_models/googlenet/` |

每个模型含 `table1_v2/`（test 用 v2 ckpt）与 `table2_v3/`（val 用 v3 ckpt）。

### 消融 8 变体（原始 test split AUC）

| 变体 | ab 码 | SA | GRN | SK-UNIT | AUC |
|------|-------|----|-----|---------|-----|
| starnet_baseline_ab000 | 000 | × | × | × | 0.952 |
| casgnet_only_skunit_ab001 | 001 | × | × | √ | 0.960 |
| casgnet_only_grn_ab010 | 010 | × | √ | × | 0.950 |
| casgnet_no_sa_ab011 | 011 | × | √ | √ | 0.954 |
| casgnet_only_sa_ab100 | 100 | √ | × | × | 0.941 |
| casgnet_no_grn_ab101 | 101 | √ | × | √ | 0.951 |
| casgnet_no_skunit_ab110 | 110 | √ | √ | × | 0.953 |
| casgnet_full_ab111 | 111 | √ | √ | √ | 0.962 |

## 5. 如何复现评估

在**原项目根目录**（含 `old_data/`、`checkpoints/` 完整路径）下：

```bash
# 从 pool 缓存重建 Version B 主表（无需重新推理）
python evaluation_results/excel_aligned/rebuild_original_split.py

# DeLong 检验（CasGNet vs 7 baselines）
python evaluation_results/excel_aligned/run_delong_tests.py

# 单模型 checkpoint 推理评估
python eval_checkpoint_on_folder.py \\
  --checkpoint checkpoints/old_data_supcon_compare_v3/casgnet_s1_ce_only/best_auc_model.pth \\
  --test-dir old_data/test
```

本发布包内脚本位于 `code/eval/`，运行前需将 `PYTHONPATH` 指向原项目根或调整路径。

## 6. 数据说明

- 划分文件：`data/splits/test_image_list.txt`（258）、`val_image_list.txt`（207）
- 图像：`data/images/` 已含 test+val 完整拷贝
- 训练集未包含；需从原项目 `old_data/train/` 获取

## 7. 文件统计

| 分区 | 文件数 | 大小 |
|------|--------|------|
| results/ | {stats.get('results_files', '?')} | {stats.get('results_size', '?')} |
| checkpoints/ | {stats.get('checkpoints_files', '?')} | {stats.get('checkpoints_size', '?')} |
| code/ | {stats.get('code_files', '?')} | {stats.get('code_size', '?')} |
| data/ | {stats.get('data_files', '?')} | {stats.get('data_size', '?')} |
| **合计** | **{stats.get('total_files', '?')}** | **{stats.get('total_size', '?')}** |

符号链接数：{stats.get('symlinks', 0)}（应为 0）

## 8. 与 PUBLICATION_PACKAGE 的关系

`results/` 是 `evaluation_results/excel_aligned/PUBLICATION_PACKAGE/` 的**完整真实拷贝**（602 个文件，约 47 MB），结构一致。本 FINAL_RELEASE 在其基础上增加了 checkpoint、数据、代码与索引文件。
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def du_sh(path: Path) -> str:
    r = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 else "?"


def count_files(path: Path) -> int:
    return sum(1 for _ in path.rglob("*") if _.is_file())


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest_entries: list = []

    # results/
    pub_src = EA / "PUBLICATION_PACKAGE"
    pub_dst = OUT / "results"
    print("Copying PUBLICATION_PACKAGE -> results/ ...")
    cp_r(pub_src, pub_dst)
    for f in ("PUBLICATION_MATERIALS_INDEX.md", "REVIEWER_ASSESSMENT.md"):
        src = EA / f
        if src.exists():
            cp(src, OUT / "results" / f)

    print("Copying checkpoints ...")
    build_checkpoints(manifest_entries)

    print("Copying data ...")
    build_data(manifest_entries)

    print("Copying code ...")
    build_code(manifest_entries)

    ckpt_idx = build_checkpoint_index()
    idx_path = OUT / "checkpoints" / "CHECKPOINT_INDEX.json"
    idx_path.write_text(json.dumps(ckpt_idx, ensure_ascii=False, indent=2), encoding="utf-8")

    build_data_readme()

    # Stats
    stats = {
        "results_files": count_files(pub_dst),
        "results_size": du_sh(pub_dst),
        "checkpoints_files": count_files(OUT / "checkpoints"),
        "checkpoints_size": du_sh(OUT / "checkpoints"),
        "code_files": count_files(OUT / "code"),
        "code_size": du_sh(OUT / "code"),
        "data_files": count_files(OUT / "data"),
        "data_size": du_sh(OUT / "data"),
        "total_files": count_files(OUT),
        "total_size": du_sh(OUT),
        "symlinks": int(subprocess.run(
            f"find {OUT} -type l | wc -l", shell=True, capture_output=True, text=True
        ).stdout.strip() or 0),
        "built_at": datetime.now().isoformat(),
    }

    build_master_readme(stats)

    # MANIFEST.json
    full_manifest = {
        "built_at": stats["built_at"],
        "source_project": str(ROOT),
        "statistics": stats,
        "files": manifest_entries,
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(full_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Verification
    pth_count = len(list(OUT.glob("checkpoints/**/best_auc_model.pth")))
    print(f"\n=== BUILD COMPLETE ===")
    print(f"Path: {OUT}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size']}")
    print(f"Symlinks: {stats['symlinks']}")
    print(f"Checkpoint .pth files: {pth_count} (expected 24: 8*2 main + 8 ablation)")
    print(f"results/ files: {stats['results_files']}")
    for sub in ("results", "checkpoints", "code", "data"):
        print(f"  {sub}/: {stats[f'{sub}_files']} files, {stats[f'{sub}_size']}")


if __name__ == "__main__":
    main()
