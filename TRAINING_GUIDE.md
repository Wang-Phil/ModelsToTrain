# 多分类任务训练指南

## 概述

`train_multiclass.py` 是一个功能完整的多分类任务训练脚本，支持：
- 多种模型架构选择
- 灵活的数据增强策略
- 多种损失函数
- 可配置的训练参数

## 快速开始

### 基础训练

```bash
D
```

## 支持的模型

### Classic Models
- `resnet50`, `resnet101`
- `inceptionv3`
- `densenet161`, `densenet201`
- `mobilenetv2`
- `googlenet`
- `efficientnetv2_s`, `efficientnetv2_m`, `efficientnetv2_l`
- `unet`, `transunet`

### ConvNeXtV2
- `convnextv2_tiny`
- `convnextv2_base`
- `convnextv2_large`
- `convnextv2_nano`

### StarNeXt
- `starnext_tiny`
- `starnext_base`
- `starnext_large`
- `starnext_small`
- `starnext_nano`

### StarNet
- `starnet_s1`
- `starnet_s2`
- `starnet_s3`
- `starnet_s4`

### MogaNet (如果可用)
- `moganet_small`
- `moganet_base`
- `moganet_large`
- `moganet_xlarge`

## 数据增强策略

### 1. `none` - 无增强
仅进行resize和normalize，适合快速测试。

### 2. `minimal` - 最小增强
基础的resize和normalize，无随机变换。

### 3. `standard` - 标准增强（推荐）
包含：
- RandomCrop
- RandomHorizontalFlip / RandomVerticalFlip
- ColorJitter
- RandomRotation
- RandomErasing

### 4. `strong` - 强增强
在标准增强基础上增加：
- 更大的旋转角度
- RandomAffine变换
- 更强的颜色抖动

### 5. `medical` - 医学图像增强
针对医学图像优化的增强策略：
- 适度的旋转和翻转
- 轻微的颜色调整
- 保持图像特征完整性

## 损失函数

### 1. `ce` - CrossEntropyLoss（默认）
标准的交叉熵损失，适用于类别平衡的数据集。

### 2. `focal` - Focal Loss
适用于类别不平衡的数据集：
```bash
--loss focal --focal-gamma 2.0
```

### 3. `label_smoothing` - Label Smoothing
提高模型泛化能力：
```bash
--loss label_smoothing --label-smoothing 0.1
```

### 4. `weighted_ce` - Weighted Cross Entropy
需要手动指定类别权重（需要在代码中修改）。

## 优化器和学习率调度

### 优化器
- `sgd`: 随机梯度下降
- `adam`: Adam优化器（默认）
- `adamw`: AdamW优化器（推荐用于Transformer类模型）

### 学习率调度器
- `cosine`: 余弦退火（推荐）
- `step`: 阶梯式衰减
- `plateau`: 基于验证损失的自动调整
- `none`: 不使用调度器

## 完整训练示例

### 示例1: ResNet50基础训练
```bash
python train_multiclass.py \
    --model resnet50 \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 50 \
    --batch-size 32 \
    --lr 0.001 \
    --optimizer adam \
    --scheduler cosine \
    --augmentation standard \
    --output-dir checkpoints/resnet50
```

### 示例2: ConvNeXtV2 + 预训练 + 强增强
```bash
python train_multiclass.py \
    --model convnextv2_tiny \
    --pretrained \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.0005 \
    --optimizer adamw \
    --scheduler cosine \
    --augmentation strong \
    --loss label_smoothing \
    --label-smoothing 0.1 \
    --output-dir checkpoints/convnextv2_strong
```

### 示例3: Focal Loss处理类别不平衡
```bash
python train_multiclass.py \
    --model efficientnetv2_s \
    --pretrained \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 80 \
    --batch-size 32 \
    --lr 0.001 \
    --loss focal \
    --focal-gamma 2.0 \
    --augmentation standard \
    --output-dir checkpoints/efficientnet_focal
```

### 示例4: 医学图像训练（推荐配置）
```bash
python train_multiclass.py \
    --model starnext_base \
    --pretrained \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 100 \
    --batch-size 24 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --loss label_smoothing \
    --label-smoothing 0.1 \
    --augmentation medical \
    --img-size 224 \
    --output-dir checkpoints/starnext_medical
```

## 参数说明

### 数据相关
- `--train-dir`: 训练数据目录（按类别组织）
- `--val-dir`: 验证数据目录
- `--img-size`: 输入图像大小（默认224）

### 模型相关
- `--model`: 模型名称
- `--pretrained`: 使用预训练权重

### 训练相关
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 初始学习率
- `--momentum`: SGD动量（默认0.9）
- `--weight-decay`: 权重衰减（默认1e-4）

### 优化器
- `--optimizer`: 优化器类型（sgd/adam/adamw）
- `--scheduler`: 学习率调度器（cosine/step/plateau/none）
- `--step-size`: StepLR的步长
- `--gamma`: StepLR的衰减率

### 损失函数
- `--loss`: 损失函数类型
- `--focal-gamma`: Focal Loss的gamma参数
- `--label-smoothing`: Label Smoothing的平滑参数

### 数据增强
- `--augmentation`: 增强类型（none/minimal/standard/strong/medical）

### 其他
- `--num-workers`: 数据加载线程数（默认4）
- `--output-dir`: 输出目录
- `--save-interval`: 保存检查点的间隔（epoch）
- `--cpu`: 强制使用CPU

## 输出文件

训练完成后，在输出目录中会生成：
- `best_model.pth`: 最佳模型（验证准确率最高）
- `checkpoint_epoch_N.pth`: 定期保存的检查点
- `history.json`: 训练历史（损失和准确率）
- `config.json`: 训练配置参数

## 模型加载

```python
import torch
from train_multiclass import create_model

# 加载最佳模型
checkpoint = torch.load('checkpoints/best_model.pth')
model = create_model(
    checkpoint['model_name'],
    num_classes=9,
    pretrained=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 注意事项

1. **内存管理**: 如果遇到内存不足，减小`--batch-size`或`--img-size`
2. **类别不平衡**: 使用`focal`损失函数或`weighted_ce`
3. **预训练权重**: 对于新模型，建议使用`--pretrained`
4. **学习率**: 使用预训练权重时，可以设置较小的学习率（如0.0001-0.001）
5. **数据增强**: 医学图像建议使用`medical`增强策略

## 故障排除

### 问题1: CUDA内存不足
- 减小`--batch-size`
- 减小`--img-size`
- 使用更小的模型

### 问题2: 训练不收敛
- 检查学习率是否过大
- 尝试使用预训练权重
- 调整数据增强策略

### 问题3: 验证准确率不提升
- 检查数据增强是否过强
- 尝试不同的损失函数
- 调整学习率调度器

## 性能优化建议

1. **数据加载**: 设置合适的`--num-workers`（通常为CPU核心数）
2. **混合精度训练**: 可以在代码中添加AMP支持
3. **分布式训练**: 对于多GPU环境，可以添加DDP支持

## 联系与支持

如有问题，请检查：
1. 数据目录结构是否正确
2. 模型名称是否正确
3. 依赖包是否安装完整

