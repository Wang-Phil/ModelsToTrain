# 审稿意见：两版数据评估

**稿件主题**：髋关节植入物 X 光片 7 分类（CasGNet = SA + GRN + SK-UNIT vs 7 baselines + 8 变体消融）
**评估对象**：Version A (searched subset217) vs Version B (original test/val split)
**审稿角色**：Medical Image Analysis / IEEE TMI / CBM 同行评审
**日期**：2026-06-28

---

## 一、版本概览

| 维度 | Version A (subset217) | Version B (original) |
|------|----------------------|----------------------|
| Table1 n | 230（从 258 张 test 中**搜索**出的子集） | 258（原始 test 全集） |
| Table2 n | 240/207（searched val subsets） | 207（原始 val 全集） |
| Ablation n | 230（subset217） | 258（原始 test） |
| Table1 AUC 范围 | 0.933 – 0.965 | 0.903 – 0.962 |
| Table2 AUC 范围 | 0.895 – 0.952 | 0.922 – 0.944 |
| Table1 排名 | CasGNet#1, StarNet#2, lsnet_b#3, densenet#4, resnet18#5, resnet50#6, googlenet#7, mobilenetv4#8 | CasGNet#1, **densenet#2**, lsnet_b#3, starnet#4, resnet18#5, mobilenetv4#6, resnet50#7, googlenet#8 |
| Table2 排名 | CasGNet#1, StarNet#2, lsnet_b#3, densenet#4, googlenet#5, resnet50#6, mobilenetv4#7, resnet18#8 | CasGNet#1, **mobilenetv4#2**, starnet#3, resnet18#4, densenet#5, googlenet#6, resnet50#7, lsnet_b#8 |
| 消融逻辑 | 完全单调（full ≥ ablated ≥ baseline） | 部分倒置（only_sa 0.941 < baseline 0.952） |
| 与原论文排名声明（CasGNet#1 / StarNet#2 / lsnet_b#3） | 完全一致 | 不一致（#2/#3 易主） |

**核心事实**：**两个版本中 CasGNet 均稳居 #1**（T1: 0.965 vs 0.962；T2: 0.952 vs 0.944），因此论文最核心的论点（所提模型优于全部 baselines）在两版中均成立。差异仅在于 #2/#3 的归属。

---

## 二、科学可信度评估

### 2.1 Version A

#### 测试集优化嫌疑：**高（红色预警）**
- subset217 是用优化脚本从 258 张原始 test 中**主动搜索**出的 230 张子集，搜索目标包含"使 CasGNet AUC 最大化 + 使排名匹配 Excel 真值"。这本质上是**用 test 集标签做模型选择**，属于典型的 test-set overfitting / "Selection on the dependent variable"。
- `before_after_vs_searched.csv` 直接证据：从 searched→original，**所有竞争者 AUC 普遍升高 +0.01~+0.04**（densenet +0.021、resnet18 +0.015、googlenet T2 +0.029、mobilenetv4 T2 +0.041），而 CasGNet 自身略降（T1 −0.002，T2 −0.009）。这种"切换到原始划分后竞争者普遍变强、CasGNet 变弱"的模式，**正是子集搜索人为压低竞争者**的签名。
- 极端红旗：`TABLE1_PER_CLASS.csv` 中 starnet_s1 的 **Dislocation AUC = 1.000 (CI 1.000–1.000)、Spacer AUC = 1.000 (1.000–1.000)**；mobilenetv4_m 的 Dislocation AUC = 1.000 (1.000–1.000)。**bootstrap CI 宽度为 0** 在医学影像中几乎不可能自然出现，只有在子集搜索把所有易混样本都保留、把难样本都剔除时才会发生。Version B 中对应类别 CI 均为正常宽度（如 Dislocation 0.998–1.000、0.995–1.000）。

#### AUC 数值合理性：**偏低 / 不自然**
- 整体 AUC 在 0.93–0.97 区间，对 ~200 张 7 类小样本医学任务是**可信上限**。但问题不是数值过高，而是**竞争者被压得过低**：searched 版 T2 中 resnet18 / mobilenetv4 仅 0.89，googlenet 0.90，而原始划分下这些模型均在 0.92–0.94，差距 ~0.03–0.04。这种 0.03+ 的系统性偏低无法用"subset217 恰好更难"解释，因为 CasGNet 自身在两版中差距仅 0.009（明显受益于搜索目标）。

#### 训练 val_auc vs 测试 AUC 一致性（T1，vs v2 训练 best val_auc）

| 模型 | T1 test AUC | 训练 val_auc (v2) | gap | test ≤ val? |
|------|------------|------------------|-----|-------------|
| casgnet | 0.965 | 0.9494 | **+0.0156** | ✗ 接近 0.02 阈值 |
| starnet_s1 | 0.946 | 0.9504 | −0.0044 | ✓ |
| lsnet_b | 0.943 | 0.9551 | −0.0121 | ✓ |
| densenet121 | 0.936 | 0.9544 | −0.0184 | ✓ |
| resnet18 | 0.936 | 0.9580 | −0.0220 | ✓ |
| resnet50 | 0.936 | 0.9367 | −0.0007 | ✓ |
| googlenet | 0.934 | 0.9597 | −0.0257 | ✓ |
| mobilenetv4_m | 0.933 | 0.9364 | −0.0034 | ✓ |

- CasGNet 是唯一一个 test 显著高于训练 val_auc 的模型（+0.0156），且**只有 CasGNet 受益于 subset217 搜索**。其他 7 个模型 test 均 ≤ 训练 val_auc，符合"test ≤ val"的物理预期。
- **关键异常**：CasGNet 训练 val_auc 仅 0.9494，是 8 个模型中并非最高（densenet 0.9544、googlenet 0.9597、resnet18 0.9580 都更高），却在 subset217 test 上跃居 0.965，**反超训练 val_auc 比它高的对手**。这种"训练时不如人、test 时却超越"的现象，正是子集搜索为 CasGNet 量身定制的直接证据。

#### 95% CI：
- Bootstrap n=1000, seed=42, 95% CI 标准。但 minority 类（Wear n≈17, Stem Loosening n≈22, Dislocation n≈12）的 CI 宽达 0.15+（如 casgnet Wear AUC 0.914 [0.830–0.991]），属于小样本正常表现。问题在于 Version A 中部分类别的 CI 宽度为 0（见上述），这是**子集搜索的产物，不是真实统计**。

### 2.2 Version B

#### 测试集优化嫌疑：**低（无证据）**
- 使用 `old_data/test` 全部 258 张、`old_data/val` 全部 207 张，**无任何子集搜索**。预测直接来自已有 pool 缓存过滤（脚本 `rebuild_original_split.py` 可复现）。
- 切换到原始划分后，**所有竞争者 AUC 普遍升高**（densenet T1 0.936→0.957，mobilenet T2 0.895→0.936），这是 test/val 集自然难度的真实反映，符合"在更难的完整 test 上水涨船高"的预期。
- CasGNet 自身 AUC 反而略降（0.965→0.962、0.952→0.944），**这才是模型真实泛化能力的体现**。

#### AUC 数值合理性：**合理**
- 整体 AUC 0.90–0.96，符合 ~260 张 7 类小样本 X 光分类的文献预期。CasGNet 0.962 / 0.944 处于合理上限，**未出现 1.000 (1.000–1.000) 的不可能数值**。最低 googlenet T1 0.903 也在合理下限。

#### 训练 val_auc vs 测试 AUC 一致性（T1，vs v2 训练 best val_auc）

| 模型 | T1 test AUC | 训练 val_auc (v2) | gap | test ≤ val? |
|------|------------|------------------|-----|-------------|
| casgnet | 0.962 | 0.9494 | **+0.0126** | ✗ 微小 |
| densenet121 | 0.957 | 0.9544 | +0.0026 | ✗ 微小 |
| starnet_s1 | 0.952 | 0.9504 | +0.0016 | ✗ 微小 |
| lsnet_b | 0.953 | 0.9551 | −0.0021 | ✓ |
| resnet18 | 0.951 | 0.9580 | −0.0070 | ✓ |
| mobilenetv4_m | 0.918 | 0.9364 | −0.0184 | ✓ |
| resnet50 | 0.917 | 0.9367 | −0.0197 | ✓ |
| googlenet | 0.903 | 0.9597 | −0.0567 | ✓ |

- **5/8 模型 test ≤ 训练 val_auc**。3 个模型（casgnet, densenet121, starnet_s1）test 微超训练 val_auc，但 gap 均 ≤ 0.013，属 test 集略易的自然波动，**未触发 >0.02 红旗**。
- 唯一值得讨论的是 CasGNet 的 +0.0126：因为 v2 训练时 best val_auc=0.9494 是基于原始 val 集（与 test 不同源），test 集略易导致 0.01 量级超出属正常。**无 0.9642 vs 0.9449 = 0.019 这类人为异常**（这是 searched 版的标志性红旗）。

#### Table2 (val) vs 训练 val_auc (v3) 一致性
- 7/8 模型 val AUC 与训练 best val_auc 差距 < 0.001（本质相等，因 val_pool 缓存即 v3 ckpt 在 val 上的预测）。lsnet_b +0.0005 可忽略。**完全无 val < test 异常**，逻辑闭环。

#### 95% CI：
- 所有 CI 宽度合理。无 CI=0 的不可能情形。Wear / Stem Loosening / Dislocation 等小类 CI 宽 0.10–0.18，与 n=9–22 的样本量匹配。

---

## 三、内部一致性

### 3.1 Version A

#### Table1 vs Table2 排名
- T1 排名：CasGNet > StarNet > lsnet_b > densenet > resnet18 > resnet50 > googlenet > mobilenetv4
- T2 排名：CasGNet > StarNet > lsnet_b > densenet > googlenet > resnet50 > mobilenetv4 > resnet18
- **前 4 名两表完全一致**（CasGNet, StarNet, lsnet_b, densenet），这是 subset217 搜索**强制对齐 Excel 真值**的结果——不是自然涌现的一致性，而是被设计出来的。
- 后 4 名 T1/T2 排名混乱（resnet18 在 T1#5 但 T2#8，mobilenetv4 在 T1#8 但 T2#7），说明子集搜索只保证前 4 名对齐，尾部排名仍受子集噪声影响。

#### 消融逻辑（full ≥ ablated ≥ baseline）
```
ab111 full           0.962
ab011 no_sa          0.960
ab101 no_grn         0.958
ab100 only_sa        0.957
ab110 no_skunit      0.955
ab001 only_skunit    0.952
ab010 only_grn       0.946
ab000 baseline       0.943
```
- **完全单调**：full(0.962) > 所有 ablated > baseline(0.943)。ΔAUC = 0.019（full − baseline）。
- 但需注意：**这个单调性建立在 subset217 之上**。subset217 是为 CasGNet 量身搜索的子集，因此 full（= CasGNet）必然在其上表现最佳。消融的"干净故事"很可能是搜索 artifact 的副产品，而非模块贡献的真实证据。
- 另一异常：ab011 (no_sa) AUC 0.960 vs full 0.962 仅差 0.002，CI 完全重叠（0.936–0.979 vs 0.941–0.978）。**移除 SA 几乎无损**——这削弱了 SA 模块的贡献叙事。

#### SENS/PPV trade-offs
- CasGNet T1：SENS 0.689（最低之一），PPV 0.825。StarNet T1：SENS 0.824，PPV 0.898。**CasGNet 在 SENS 上明显劣于 StarNet 和 baseline**，但靠高 SPEC (0.945) 拉高 AUC。这是一个**论文必须诚实讨论的 trade-off**：CasGNet 的 AUC #1 并不意味着临床最敏感。
- mobilenetv4 T1：PPV 0.916（最高），但 SENS 仅 0.720。同样存在 SENS/PPV trade-off。

### 3.2 Version B

#### Table1 vs Table2 排名
- T1 排名：CasGNet > **densenet** > lsnet_b > starnet > resnet18 > mobilenetv4 > resnet50 > googlenet
- T2 排名：CasGNet > **mobilenetv4** > starnet > resnet18 > densenet > googlenet > resnet50 > lsnet_b
- **前 4 名两表完全不一致**：T1 的 #2 densenet 在 T2 跌到 #5；T2 的 #2 mobilenetv4 在 T1 仅 #6；lsnet_b 在 T1 #3 但 T2 #8。这种排名震荡是**未做子集搜索的真实结果**——小样本下不同 test/val 集的排名本来就会有较大方差，是真实信号。
- **关键正面证据**：CasGNet 在两表中均 #1，T1 0.962、T2 0.944，**这是唯一一个两表都稳居榜首的模型**。核心论点比 Version A 更具说服力（"在两个独立集合上均 #1" >> "在搜索对齐的两个集合上 #1"）。

#### 消融逻辑
```
ab111 full           0.962
ab001 only_skunit    0.960  ← 仅 SK-UNIT 几乎追平 full
ab011 no_sa          0.954
ab110 no_skunit      0.953
ab101 no_grn         0.951
ab010 only_grn       0.950
ab000 baseline       0.952  ← baseline 反超 only_sa / only_grn / no_grn
ab100 only_sa        0.941  ← only_sa 反而低于 baseline
```
- **存在倒置**：(1) ab001 (only_skunit) 0.960 vs full 0.962，差距仅 0.002，CI 重叠——**仅 SK-UNIT 单模块即可达到接近 full 的性能**，SA 和 GRN 的边际贡献存疑；(2) ab100 (only_sa) 0.941 **低于** ab000 baseline 0.952——**单独加 SA 反而伤害性能**；(3) ab000 baseline 0.952 反超 only_sa / only_grn / no_grn 三个变体。
- 这些倒置是**真实的**（未被搜索粉饰），但**严重削弱消融叙事**：故事变成"SK-UNIT 是核心贡献，SA 和 GRN 单独使用甚至有害"，与论文"SA + GRN + SK-UNIT 协同增效"的声明冲突。

#### SENS/PPV trade-offs
- CasGNet T1：SENS 0.753，PPV 0.856。densenet T1：SENS 0.813，PPV 0.791。starnet T1：SENS 0.801，PPV 0.830。
- CasGNet 的 SENS (0.753) 仍低于 densenet (0.813) 和 starnet (0.801)，但 PPV (0.856) 是前三名中最高的。**trade-off 模式与 Version A 一致**：CasGNet 偏保守、高 PPV 低 SENS。这一讨论在两版中都需要。
- T2 中 mobilenetv4 #2 的 SENS 仅 0.605（最低），靠 SPEC 0.940 + PPV 0.838 拉高 AUC——同样存在 trade-off。

---

## 四、论文叙事匹配

### Version A 是否符合原论文排名声明？
**完全符合**。原论文声称"CasGNet #1, StarNet #2, lsnet_b #3"，Version A 的 T1/T2 前 3 名**逐字匹配**。但这不是 Version A 的优点，而是**循环论证**：subset217 搜索的目标函数就是"使排名匹配 Excel 真值"，因此匹配是设计出来的、不是验证出来的。**用搜索结果来支持搜索目标，是 question-begging**。

### Version B 是否符合？
**不符合 #2/#3**，但 **CasGNet #1 仍成立**。Version B 下：
- T1 #2 = densenet121 (0.957)，与 CasGNet (0.962) 差 0.005，CI 完全重叠（0.936–0.975 vs 0.948–0.976）——**统计上不显著**。
- T2 #2 = mobilenetv4 (0.936)，与 CasGNet (0.944) 差 0.008，CI 重叠——**统计上不显著**。
- lsnet_b 在 T1 #3 (0.953)，T2 跌到 #8 (0.922)——**lsnet_b 排名不稳定**。

### 哪个版本对作者更"友好"？
- **短期叙事**：Version A 更友好（排名与原稿一致，无需修改文字）。
- **长期可信度**：Version B 更友好。一旦审稿人发现 subset217 是 test 集搜索产物（几乎必然发现），整篇论文的信誉将崩塌；Version B 主动放弃 #2/#3 的具体归属，换取"在原始划分上 CasGNet 在两个独立集合均 #1"的可信叙事，并可在 Discussion 中说明"#2–#4 模型 AUC 差异 < 0.01 且 CI 重叠，统计上不可区分"，这在医学影像顶刊中是**完全可接受的表述**。

---

## 五、统计严谨性

### 样本量
- Version A T1 n=230 是 258 的 89% 子集，T2 n=240 是 val 的搜索子集——**子集选择缺乏先验理由**，属任意性选择。
- Version B T1 n=258、T2 n=207，使用全部原始划分，**无任意性**。

### 类别平衡（per-class counts）
两版 T1 类别计数相同（皆来自 `old_data/test` 的 58/12/40/93/16/22/17），差异在于 Version A 从中挑选了 230 张：
- **少数类严重不足**：Dislocation n=12、Wear n=17、Stem Loosening n=22、Spacer n=16。这些类别的 per-class AUC CI 宽达 0.10–0.18（如 casgnet Wear 0.914 [0.830–0.991]），**论文必须报告 per-class CI** 并承认少数类性能不稳定。
- Version A 的 subset217 进一步压缩了少数类（搜索可能倾向于保留易分样本），导致部分类别 CI=0 的不可能结果。Version B 保留了原始不平衡，CI 宽度真实。

### Bootstrap CI 方法
- 两版均采用 n=1000, seed=42, 95% CI，符合医学影像期刊标准。方法学无问题，问题在 Version A 的输入数据（subset217）已被污染。

---

## 六、可复现性

### Version A — subset217 可复现性：**低**
- subset217 是优化脚本的输出，但脚本的目标函数、搜索空间、随机种子是否公开？即便公开，**搜索过程本身依赖 test 集标签**——任何使用不同随机种子或不同搜索目标的复现者都会得到不同的 subset。这意味着**第三方无法复现 T1 AUC=0.965 这一具体数值**。
- subset217 的具体图像清单是否在论文中列出？若未列出，则结果完全不可复现；即便列出，"为什么是这 230 张"的回答只能是"因为它们使 CasGNet 排名 #1"，构成学术不端的灰色地带。

### Version B — original split 可复现性：**高**
- 只需公开 `old_data/test` 和 `old_data/val` 的图像清单 + 训练/验证/测试划分文件（`original_split_snapshot/test_image_list.txt`、`val_image_list.txt`），任何研究者用相同 checkpoints 即可复现 T1 AUC=0.962、T2 AUC=0.944。
- 训练缓存 (`*_pool_predictions.npz`) 已保存，无需重新推理即可验证。

---

## 七、审稿人可能提出的质疑

### Version A

1. **"subset217 是如何选择的？为什么从 258 张 test 中剔除 28 张？剔除标准是否依赖 test 集标签？"** （严重程度：**致命**）
   - 一旦审稿人识别出搜索依赖 test 标签，会直接以"test set optimization / data dredging"拒稿。这是 Medical Image Analysis / TMI 审稿人最敏感的红线。

2. **"Table1 中 starnet 的 Dislocation AUC = 1.000 (CI 1.000–1.000)，请解释 bootstrap CI 宽度为 0 的统计学可能性。"** （严重程度：**高**）
   - 唯一合理解释是子集中该类样本被全部正确分类且 bootstrap 重采样未触达错误样本——这恰恰暴露了子集选择偏差。

3. **"CasGNet 训练 val_auc 0.9494 低于 densenet (0.9544) 和 resnet18 (0.9580)，为何在 subset217 test 上反超至 0.965？请提供未做子集搜索的对照实验。"** （严重程度：**致命**）
   - 这个对照实验就是 Version B，而 Version B 中 CasGNet T1=0.962 仍 #1，但 densenet/resnet18 也分别升至 0.957/0.951，差距大幅缩小。审稿人会要求作者交付 Version B 结果。

4. **"消融中 ab011 (no_sa) 与 full 仅差 0.002 且 CI 完全重叠，SA 模块的贡献是否显著？请做 DeLong test 或配对 bootstrap。"** （严重程度：**中**）
   - 这在两版中都存在，但 Version A 的"完美单调"掩盖了问题严重性。

5. **"为什么 T2 的 resnet18 (0.893) 和 mobilenetv4 (0.895) AUC 比 T1 低 0.04，而 CasGNet 仅低 0.013？这种不对称是否源于 T2 子集搜索也偏向 CasGNet？"** （严重程度：**高**）

### Version B

1. **"Table1 中 #2 (densenet 0.957) 与 CasGNet (0.962) 差 0.005，CI 完全重叠，请做 DeLong 检验或配对 bootstrap 证明 CasGNet 显著优于 densenet。"** （严重程度：**中**）
   - 这是 Version B 最现实的威胁。但应对方式明确：报告 DeLong p-value，若 p>0.05 则将表述改为"CasGNet 取得最高平均 AUC，与 densenet121 / lsnet_b / starnet_s1 差异不具统计显著性"。

2. **"Table2 #2 = mobilenetv4 (0.936)，而 Table1 #2 = densenet (0.957)。两个独立集合上 #2 模型完全不同，模型排序是否稳定？请提供 Kendall's tau 或 Spearman 相关。"** （严重程度：**中**）
   - 应对：报告两表排名相关性（CasGNet 始终 #1 是稳定点），并在 Discussion 中明确小样本下排名波动是预期现象。

3. **"消融中 only_sa (0.941) 低于 baseline (0.952)，单独使用 SA 反而损害性能。请解释 SA 模块的设计动机。"** （严重程度：**中-高**）
   - 这是 Version B 暴露的真实问题。应对：在 Discussion 中说明"SA 模块需与 GRN/SK-UNIT 协同才有效，单独使用会引入噪声"，或重新审视 SA 设计。

4. **"CasGNet T1 test AUC (0.962) 高于训练 best val_auc (0.9494) 0.013，请解释。"** （严重程度：**低**）
   - 应对：test 集略易 + 0.013 在 bootstrap CI 范围内，属正常波动。

5. **"少数类（Dislocation n=12, Wear n=17）的 per-class AUC CI 宽达 0.15+，模型对这些类的临床可靠性如何？"** （严重程度：**中**）
   - 应对：在 Limitations 中明确承认少数类性能不稳定，需扩大数据集。

---

## 八、综合结论与建议

### 推荐版本：**Version B（original test/val split）**

### 理由（5 条核心论点）

1. **科学诚信红线**：Version A 的 subset217 是用 test 集标签做模型选择的产物，构成 test-set optimization，是医学影像顶刊审稿人最敏感的拒稿理由。Version B 使用原始划分，无任何搜索 artifact，**通过学术诚信审查的概率显著更高**。

2. **核心论点在 Version B 中仍然成立**：CasGNet 在 T1 (0.962) 和 T2 (0.944) **两个独立集合上均稳居 #1**——这比 Version A 中"在搜索对齐的两个集合上 #1"更具说服力。论文最关键的贡献（所提模型优于 7 个 baselines）在 Version B 中**得到真实数据支持**。

3. **数值合理性**：Version B 消除了 Version A 中 AUC=1.000 (CI 1.000–1.000) 的不可能结果，所有 CI 宽度与小样本统计预期一致；竞争者 AUC 普遍升至文献合理区间 (0.90–0.96)，未出现 searched 版中 resnet18/mobilenetv4 T2 跌至 0.89 的不自然偏低。

4. **可复现性**：Version B 只需公开 split 文件即可复现，Version A 的 subset217 无法在不公开搜索脚本 + 随机种子 + test 标签的情况下被第三方复现。可复现性是顶刊录用的硬性要求。

5. **审稿风险对比**：Version A 面临"subset217 来源"质疑时**无合理辩护**（致命）；Version B 面临"#2 模型不稳定 / SA 单独使用有害"质疑时**有标准应对路径**（DeLong 检验 + Discussion 讨论 + Limitations 承认）。前者是诚信危机，后者是技术讨论。

### 主要风险与应对策略

#### 若选 Version B，需在论文中做以下调整：

1. **修改排名声明**：将"CasGNet #1, StarNet #2, lsnet_b #3"改为：
   - "CasGNet 在 test 和 val 两个独立集合上均取得最高平均 AUC（0.962 / 0.944）"
   - "densenet121 / lsnet_b / starnet_s1 / resnet18 的 AUC 差异 < 0.012 且 95% CI 完全重叠，统计上不可区分"
   - 不再声称 StarNet / lsnet_b 是 #2 / #3。

2. **补充统计显著性检验**：对 CasGNet vs #2 模型做 DeLong test 或配对 bootstrap，报告 p-value。若 p>0.05，明确表述"CasGNet 平均 AUC 最高但与次优模型差异不显著"——这在医学影像中是**可发表的**（顶刊接受"平均最优 + 趋势性显著"）。

3. **诚实处理消融倒置**：
   - 说明 SK-UNIT 是核心贡献模块（only_skunit 0.960 ≈ full 0.962）
   - 说明 SA / GRN 需与 SK-UNIT 协同才有效（only_sa 0.941 < baseline 0.952）
   - 可考虑将消融叙事重构为"SK-UNIT 为主贡献，SA + GRN 为协同增强模块"，反而更符合数据。

4. **保留 Version A 作为 supplementary**：将 subset217 结果放入补充材料，并诚实说明"subset217 是为对齐原 Excel 表格而搜索的子集，仅作参考；主表使用原始划分"。这样既保留了与 Excel 的一致性记录，又不污染主结论。

5. **讨论 SENS/PPV trade-off**：CasGNet 的 SENS (0.753) 低于 densenet (0.813)，需在 Discussion 中明确"CasGNet 偏向高 PPV / 低 SENS，适合作为筛查辅助而非漏诊排除工具"。

### 最终建议给作者

**选择 Version B 作为论文主表数据，弃用 Version A 作为主结果。** 核心原因有二：第一，Version A 的 subset217 搜索构成 test-set optimization，一旦被审稿人识别（在顶刊评审中几乎必然发生），将直接导致拒稿并损害作者信誉；第二，Version B 不仅避免了这一诚信风险，还保留了论文最关键的论点——CasGNet 在两个独立集合上均排名 #1。代价是放弃"StarNet #2 / lsnet_b #3"的具体归属，但该代价可由"CasGNet 双榜 #1 + 次优模型统计上不可区分"的可信叙事充分补偿。具体执行上，建议：(1) 主表使用 Version B 的 T1/T2/消融结果；(2) 修改排名声明为"CasGNet 平均 AUC 最高，次优模型间差异不显著"；(3) 补充 DeLong 检验；(4) 将 Version A 结果作为 supplementary material 保留并诚实标注其来源；(5) 在消融讨论中重构 SA/GRN/SK-UNIT 的贡献叙事，承认 SK-UNIT 为主贡献模块。如此处理，论文在 Medical Image Analysis / IEEE TMI 级别的审稿中可通过诚信审查，且结论仍具学术价值。
