# CLIP模型训练指南

## 概述

本指南介绍如何使用 `train_clip.py` 训练CLIP风格的医学图像分类模型。该训练脚本支持多个图像编码器和文本编码器的组合训练。

## 模型架构

CLIP模型包含两个主要组件：
- **图像编码器**: 将图像编码为特征向量
- **文本编码器**: 将文本描述编码为特征向量

支持的图像编码器：
- `starnet_dual_pyramid_rcf` - StarNet双金字塔RCF模型
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` - ResNet系列
- `vit` - Vision Transformer
- `efficientnet-b0` 到 `efficientnet-b7` - EfficientNet系列
- `convnext-tiny`, `convnext-small`, `convnext-base`, `convnext-large` - ConvNeXt系列

支持的文本编码器：
- `bert-base-chinese` - 中文BERT模型

## 快速开始

### 1. 单配置训练

训练单个模型配置：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --embed-dim 512 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --gpu-id 0
```

### 2. 五折交叉验证训练

使用交叉验证训练，获得更可靠的模型性能评估：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/cv \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --use-cv \
    --n-splits 5 \
    --batch-size 32 \
    --epochs 100 \
    --gpu-id 0
```

### 3. 多配置训练

使用配置文件训练多个模型组合：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models \
    --multi-config \
    --config-file train_clip_config.json \
    --gpu-id 0
```

### 4. 使用Shell脚本

使用提供的Shell脚本进行训练：

```bash
bash run_train_clip.sh
```

## 参数说明

### 数据参数
- `--data-dir`: 数据目录路径（按类别组织的文件夹结构）
- `--output-dir`: 输出目录路径

### 文本描述参数
- `--text-template`: 文本模板，例如 `"这是一张{class_name}的图片"`，默认使用类别名称
- `--class-texts-file`: 类别文本描述JSON文件路径，格式: `{"类别1": "描述1", "类别2": "描述2"}`

### 模型参数
- `--image-encoder`: 图像编码器名称（默认: `starnet_dual_pyramid_rcf`）
- `--text-encoder`: 文本编码器名称（默认: `bert-base-chinese`）
- `--embed-dim`: 嵌入维度（默认: 512）
- `--temperature`: 温度参数（默认: 0.07）

### 训练参数
- `--batch-size`: 批次大小（默认: 32）
- `--epochs`: 训练轮数（默认: 100）
- `--learning-rate`: 学习率（默认: 1e-4）
- `--weight-decay`: 权重衰减（默认: 0.01）
- `--img-size`: 图像大小（默认: 224）
- `--augmentation`: 数据增强类型，可选: `none`, `minimal`, `standard`（默认: `standard`）

### 其他参数
- `--num-workers`: 数据加载工作进程数（默认: 4）
- `--gpu-id`: GPU ID（默认: 0）
- `--no-amp`: 禁用混合精度训练
- `--resume-from`: 恢复训练的checkpoint路径
- `--no-save-best`: 不保存最佳模型

### 交叉验证参数
- `--use-cv`: 使用K折交叉验证训练
- `--n-splits`: 交叉验证折数（默认: 5）
- `--random-state`: 随机种子（默认: 42）

### 早停参数
- `--early-stopping-patience`: 早停耐心值，连续多少个epoch没有改善就停止训练（默认: None，即不使用早停）
- `--early-stopping-min-delta`: 早停最小改进阈值，只有当改进超过这个值才算改善（默认: 0.0）
- `--early-stopping-monitor`: 早停监控指标，可选 `val_acc`（验证准确率）或 `val_loss`（验证损失），默认 `val_acc`

### 多配置训练参数
- `--multi-config`: 启用多配置训练模式
- `--config-file`: 配置文件路径（JSON格式）

## 配置文件格式

配置文件是一个JSON数组，每个元素是一个配置字典：

```json
[
  {
    "image_encoder_name": "starnet_dual_pyramid_rcf",
    "text_encoder_name": "bert-base-chinese",
    "embed_dim": 512,
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "temperature": 0.07,
    "img_size": 224,
    "augmentation": "standard",
    "num_workers": 4,
    "use_amp": true,
    "gpu_id": 0,
    "save_best": true,
    "use_cv": false,
    "n_splits": 5,
    "early_stopping_patience": null,
    "early_stopping_min_delta": 0.1,
    "early_stopping_monitor": "val_acc"
  },
  {
    "image_encoder_name": "resnet50",
    "text_encoder_name": "bert-base-chinese",
    ...
  }
]
```

## 数据格式

数据目录应该是按类别组织的文件夹结构：

```
single_label_data/
├── 类别1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 类别2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

## 类别文本描述

CLIP模型通过图像-文本对齐来学习，因此类别文本描述对模型性能非常重要。

### 方法1: 使用文本模板（简单）

使用统一的模板为所有类别生成描述：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --text-template "这是一张{class_name}的医学图像" \
    ...
```

### 方法2: 使用JSON配置文件（推荐）

为每个类别自定义详细的文本描述：

**步骤1：生成类别文本描述文件**

```bash
python generate_class_texts.py \
    --data-dir single_label_data \
    --output class_texts.json \
    --template "这是一张{class_name}的医学图像"
```

或者交互式生成（为每个类别输入自定义描述）：

```bash
python generate_class_texts.py \
    --data-dir single_label_data \
    --output class_texts.json \
    --interactive
```

**步骤2：编辑生成的JSON文件**

编辑 `class_texts.json`，为每个类别添加详细的描述：

```json
{
  "正常": "这是一张显示正常解剖结构的医学X光图像，组织清晰，没有异常阴影",
  "异常": "这是一张显示异常病变的医学X光图像，可见异常阴影或病变区域",
  "肺炎": "这是一张显示肺炎病变的胸部X光图像，可见肺部炎症阴影"
}
```

**步骤3：使用配置文件训练**

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --class-texts-file class_texts.json \
    ...
```

### 方法3: 在配置文件中指定

在 `train_clip_config.json` 中添加：

```json
{
  "image_encoder_name": "starnet_dual_pyramid_rcf",
  "text_encoder_name": "bert-base-chinese",
  "text_template": "这是一张{class_name}的医学图像",
  "class_texts_file": "class_texts.json",
  ...
}
```

更多详细信息请参考 `CLASS_TEXTS_GUIDE.md`。

## 输出文件

### 普通训练模式

训练完成后，输出目录将包含：

- `config.json`: 训练配置
- `checkpoint_latest.pth`: 最新checkpoint
- `checkpoint_best.pth`: 最佳模型checkpoint（如果启用）
- `history.json`: 训练历史记录

### 交叉验证模式

交叉验证训练完成后，输出目录将包含：

- `config.json`: 训练配置
- `cv_summary.json`: 交叉验证汇总结果（包含所有fold的平均指标）
- `fold_1/`, `fold_2/`, ..., `fold_5/`: 每个fold的训练结果目录
  - `checkpoint_latest.pth`: 最新checkpoint
  - `checkpoint_best.pth`: 最佳模型checkpoint
  - `history.json`: 训练历史记录

`cv_summary.json` 包含：
- 所有fold的详细结果
- 平均指标（mean ± std）：
  - 训练损失和准确率
  - 验证损失和准确率
  - 最佳验证准确率

Checkpoint包含：
- 模型权重
- 优化器状态
- 学习率调度器状态
- 训练历史
- 类别映射
- 类别文本描述

## 恢复训练

从checkpoint恢复训练：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --resume-from checkpoints/clip_models/my_model/checkpoint_latest.pth \
    --gpu-id 0
```

## 测试模型

在训练之前，可以使用测试脚本验证模型是否能正常工作：

```bash
python test_clip_model.py
```

这将测试：
- 图像编码器的创建和前向传播
- 文本编码器的创建和前向传播
- 完整CLIP模型的创建和前向传播

## 训练监控

训练过程中会显示：
- 每个epoch的训练损失和准确率
- 验证损失和准确率
- 最佳验证准确率

训练历史保存在 `history.json` 文件中，可以使用以下代码可视化：

```python
import json
import matplotlib.pyplot as plt

with open('checkpoints/clip_models/my_model/history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
```

## 早停策略

早停策略可以在模型性能不再提升时自动停止训练，节省训练时间并防止过拟合。

### 工作原理

早停策略会监控验证集指标（验证准确率或验证损失）：
- 如果监控指标连续 `patience` 个epoch没有改善，训练将自动停止
- `min_delta` 参数可以设置最小改进阈值，只有当改进超过这个阈值才算改善
- 支持监控 `val_acc`（验证准确率，越高越好）或 `val_loss`（验证损失，越低越好）

### 使用示例

```bash
# 监控验证准确率，连续30个epoch没有提升0.1%以上就停止
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --early-stopping-patience 30 \
    --early-stopping-min-delta 0.1 \
    --early-stopping-monitor val_acc \
    ...
```

### 配置文件设置

在配置文件中可以设置：

```json
{
  "early_stopping_patience": 30,
  "early_stopping_min_delta": 0.1,
  "early_stopping_monitor": "val_acc"
}
```

如果不设置 `early_stopping_patience` 或设置为 `null`，则不会使用早停策略。

## 常见问题

### 1. 内存不足
- 减小 `--batch-size`
- 使用较小的图像尺寸 `--img-size 224`
- 禁用混合精度训练 `--no-amp`

### 2. 训练速度慢
- 增加 `--num-workers`
- 使用混合精度训练（默认启用）
- 使用更小的模型

### 3. 模型不收敛
- 降低学习率
- 增加训练轮数
- 调整温度参数
- 使用数据增强

### 4. 无法加载BERT模型
脚本会自动尝试使用Hugging Face镜像站点。如果仍然失败：
- 检查网络连接
- 设置代理环境变量
- 手动下载模型到本地

## 示例

### 示例1: 训练StarNet + BERT模型

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/starnet_bert \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --gpu-id 0
```

### 示例2: 五折交叉验证训练（带早停）

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/cv_starnet \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --use-cv \
    --n-splits 5 \
    --batch-size 32 \
    --epochs 100 \
    --early-stopping-patience 30 \
    --early-stopping-min-delta 0.1 \
    --early-stopping-monitor val_acc \
    --gpu-id 0
```

### 示例2b: 使用早停策略

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/my_model \
    --image-encoder starnet_dual_pyramid_rcf \
    --text-encoder bert-base-chinese \
    --batch-size 32 \
    --epochs 100 \
    --early-stopping-patience 30 \
    --early-stopping-min-delta 0.1 \
    --early-stopping-monitor val_acc \
    --gpu-id 0
```

早停策略说明：
- `--early-stopping-patience 30`: 如果连续30个epoch验证准确率没有改善，则停止训练
- `--early-stopping-min-delta 0.1`: 只有当准确率提升超过0.1%才算改善
- `--early-stopping-monitor val_acc`: 监控验证准确率（也可以使用 `val_loss` 监控验证损失）

### 示例3: 训练多个模型组合

编辑 `train_clip_config.json`，然后运行：

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models \
    --multi-config \
    --config-file train_clip_config.json \
    --gpu-id 0
```

在配置文件中，可以设置 `"use_cv": true` 和 `"n_splits": 5` 来为每个配置启用交叉验证。

## 参考

- CLIP论文: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- StarNet模型: `models/starnet_dual_pyramid_rcf.py`
- CLIP模型实现: `models/clip.py`

