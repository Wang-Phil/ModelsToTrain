# LSAL (LLM-Semantic Adaptive Loss) 使用指南

## 概述

LSAL (LLM-Semantic Adaptive Loss) 是一种创新的损失函数，专门为医学图像分类设计。它利用LLM（大语言模型）的医学知识来构建类别间的语义相似度矩阵，从而：

1. **防止过拟合**：使用软标签替代硬编码的One-hot标签，增加训练噪声和鲁棒性
2. **保留显式监督**：保留SuperCLIP的核心分类损失，但升级为LLM认知的软标签
3. **语义锚点约束**：强制图像特征向LLM定义的"语义中心"靠拢，防止特征偏移

## 核心创新

### 1. LLM-Guided Soft Target Cross Entropy (L_LSA-CE)

将One-hot标签替换为软标签，基于LLM生成的语义相似度矩阵：

$$L_{LSA-CE} = - \sum_{c=1}^{C} Y_{soft}^{(c)} \log(\text{Softmax}(Logits)^{(c)})$$

其中 $Y_{soft} = M[y]$，$M$ 是LLM语义相似度矩阵。

### 2. Semantic Anchor Loss (L_Anchor)

强制图像特征向该类别的LLM描述特征中心对齐：

$$L_{Anchor} = \| I - T_{center}^y \|^2$$

## 使用步骤

### 第一步：生成LLM语义矩阵（离线运行一次）

在训练开始前，需要先生成语义矩阵和类别中心：

```bash
# 方法1：从命令行指定类别名称
python models/build_llm_semantics.py \
    --classnames "Pneumonia" "Fracture" "Edema" "Atelectasis" \
    --output_dir ./semantics \
    --tau 0.1 \
    --device cuda

# 方法2：从JSON文件加载类别名称
python models/build_llm_semantics.py \
    --classnames_file ./classnames.json \
    --output_dir ./semantics \
    --tau 0.1 \
    --device cuda
```

**参数说明：**
- `--classnames`: 类别名称列表（空格分隔）
- `--classnames_file`: 包含类别名称的JSON文件（格式：`{"0": "Pneumonia", "1": "Fracture", ...}`）
- `--output_dir`: 输出目录，将保存以下文件：
  - `class_centers.pt`: 类别中心 [N_classes, Dim]
  - `soft_labels_matrix.pt`: 软标签矩阵 [N_classes, N_classes]
  - `classnames.json`: 类别名称映射
  - `config.json`: 配置信息
- `--tau`: 温度系数（默认0.1），越小越接近One-hot，越大越平滑
- `--device`: 计算设备（cuda/cpu）

**输出文件：**
```
semantics/
├── class_centers.pt          # 类别中心特征
├── soft_labels_matrix.pt     # 软标签矩阵
├── classnames.json           # 类别名称映射
└── config.json               # 配置信息
```

### 第二步：配置训练

在Dassl配置文件中，需要添加LSAL相关设置：

```python
# 在配置文件中添加
cfg.TRAINER.NAME = "LSAL_BiomedCLIP"
cfg.TRAINER.LSAL.PREC = "amp"  # 或 "fp32", "fp16"
cfg.TRAINER.LSAL.SEMANTICS_DIR = "./semantics"  # 语义文件目录
cfg.TRAINER.LSAL.LAMBDA_ANCHOR = 0.5  # Semantic Anchor Loss的权重
```

**配置参数说明：**
- `PREC`: 精度设置（"amp"混合精度, "fp32"全精度, "fp16"半精度）
- `SEMANTICS_DIR`: 语义文件目录路径（或使用`SEMANTICS_FILE`指定单个文件路径）
- `LAMBDA_ANCHOR`: Semantic Anchor Loss的权重系数（默认0.5）

### 第三步：运行训练

使用标准的Dassl训练命令：

```bash
python train.py \
    --root ./data \
    --trainer LSAL_BiomedCLIP \
    --dataset-config-file configs/datasets/your_dataset.yaml \
    --config-file configs/trainers/LSAL_BiomedCLIP/your_config.yaml
```

## 代码结构

```
models/
├── build_llm_semantics.py    # 离线生成语义矩阵的脚本
├── lsal_biomedclip.py        # LSAL Trainer和Loss实现
└── LSAL_README.md            # 本文档
```

## 核心类说明

### `LLMSemanticSuperLoss`

损失函数类，包含两个部分：
- **LLM-Guided Soft Target Cross Entropy**: 使用软标签的交叉熵损失
- **Semantic Anchor Loss**: 图像特征与类别中心的MSE损失

### `LSAL_BiomedCLIP`

Trainer类，继承自`TrainerX`：
- 只训练图像编码器（Visual Encoder）
- 冻结文本编码器，使用预计算的类别中心
- 使用LSAL损失函数进行训练

## 优势

1. **防止过拟合**：软标签增加了训练噪声，提高模型鲁棒性
2. **医学知识注入**：利用LLM的医学知识构建类别相似度
3. **特征稳定性**：语义锚点损失防止特征在微调时偏移
4. **极简架构**：无Prompt Learning，显存占用小，训练速度快

## 超参数调优建议

1. **tau (温度系数)**：
   - 较小值（0.05-0.1）：更接近One-hot，适合类别差异明显的任务
   - 较大值（0.2-0.5）：更平滑，适合类别相似度高的任务

2. **lambda_anchor (锚点损失权重)**：
   - 较小值（0.1-0.3）：更关注分类准确性
   - 较大值（0.5-1.0）：更强调特征对齐

## 注意事项

1. **类别顺序**：确保训练时的类别顺序与生成语义矩阵时的顺序一致
2. **类别数量**：如果数据集类别数量与语义矩阵不匹配，代码会自动处理（截断或报错）
3. **设备一致性**：生成语义矩阵时使用的设备不影响训练（会自动转换）

## 示例：完整工作流

```bash
# 1. 生成语义矩阵
python models/build_llm_semantics.py \
    --classnames "Pneumonia" "Fracture" "Edema" \
    --output_dir ./semantics \
    --tau 0.1

# 2. 训练模型（假设已有配置文件）
python train.py \
    --root ./data \
    --trainer LSAL_BiomedCLIP \
    --config-file configs/trainers/LSAL_BiomedCLIP/vit_b16.yaml
```

## 故障排除

1. **找不到语义文件**：
   - 检查`SEMANTICS_DIR`路径是否正确
   - 确保已运行`build_llm_semantics.py`生成文件

2. **类别数量不匹配**：
   - 重新生成语义矩阵，确保类别名称和顺序与数据集一致

3. **显存不足**：
   - 使用`PREC="amp"`启用混合精度训练
   - 减小batch size

## 参考文献

- SuperCLIP: 显式监督的CLIP训练
- Label Smoothing: 软标签技术
- Metric Learning: 度量学习（语义锚点损失）

