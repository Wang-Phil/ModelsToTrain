# FINAL_RELEASE — 论文最终结果完整发布包

**生成日期**: 2026-07-04  
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
python eval_checkpoint_on_folder.py \
  --checkpoint checkpoints/old_data_supcon_compare_v3/casgnet_s1_ce_only/best_auc_model.pth \
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
| results/ | 602 | 47M |
| checkpoints/ | 49 | 767M |
| code/ | 22 | 484K |
| data/ | 492 | 91M |
| **合计** | **1165** | **905M** |

符号链接数：0（应为 0）

## 8. 与 PUBLICATION_PACKAGE 的关系

`results/` 是 `evaluation_results/excel_aligned/PUBLICATION_PACKAGE/` 的**完整真实拷贝**（602 个文件，约 47 MB），结构一致。本 FINAL_RELEASE 在其基础上增加了 checkpoint、数据、代码与索引文件。
