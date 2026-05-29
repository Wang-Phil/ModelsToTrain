# CLIP 模型训练指南

## 概述

本训练脚本支持使用**五折交叉验证**训练 CLIP 模型，支持多种预训练的图像编码器和文本编码器。

## 快速开始

### 方法一：使用 Shell 脚本（推荐）

1. **编辑配置文件 `run_train_clip.sh`**：

```bash
# 设置数据目录
DATA_DIR="single_label_data"

# 设置 GPU
GPUS=(6)  # 使用 GPU 6

# 设置训练参数
BATCH_SIZE=16
EPOCHS=200
LEARNING_RATE=1e-4

# 启用交叉验证
USE_CV=true
N_SPLITS=5

# 选择图像编码器
IMAGE_ENCODERS=(
    "resnet18"
    # "starnet_s1:pretrained"  # 使用预训练权重
)

# 选择文本编码器
TEXT_ENCODERS=(
    "clip:ViT-B/32"
    # "bert-base-chinese"  # 中文文本编码器
)
```

2. **运行训练**：

```bash
bash run_train_clip.sh
```

### 方法二：使用配置文件

1. **编辑 `train_clip_config.json`**：

```json
[
  {
    "image_encoder_name": "resnet18",
    "text_encoder_name": "clip:ViT-B/32",
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
    "use_cv": true,
    "n_splits": 5,
    "random_state": 42,
    "class_texts_file": "class_texts_hip_prosthesis.json",
    "use_weighted_sampling": true,
    "weight_method": "inverse_freq",
    "weight_smooth_factor": 1.0
  }
]
```

2. **在 `run_train_clip.sh` 中设置**：

```bash
USE_MULTI_CONFIG=true
CONFIG_FILE="train_clip_config.json"
```

3. **运行训练**：

```bash
bash run_train_clip.sh
```

### 方法三：直接使用 Python 脚本

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/resnet18_clip_ViT-B_32 \
    --image-encoder resnet18 \
    --text-encoder clip:ViT-B/32 \
    --embed-dim 512 \
    --batch-size 16 \
    --epochs 200 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --temperature 0.07 \
    --img-size 224 \
    --augmentation standard \
    --num-workers 4 \
    --gpu-id 6 \
    --use-cv \
    --n-splits 5 \
    --random-state 42 \
    --class-texts-file class_texts_hip_prosthesis.json \
    --use-weighted-sampling \
    --weight-method inverse_freq
```

## 支持的模型

### 图像编码器

#### ResNet 系列
- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`

#### StarNet 系列（有预训练权重）
- `starnet_s1` - 2.9M 参数，最快
- `starnet_s2` - 3.7M 参数，平衡
- `starnet_s3` - 5.8M 参数，更高精度
- `starnet_s4` - 7.5M 参数，最高精度
- `starnet_s050`, `starnet_s100`, `starnet_s150` - 超小模型

**使用预训练权重**：
- `starnet_s1:pretrained` 或 `starnet_s1:true`

#### EfficientNet 系列
- `efficientnet-b0` 到 `efficientnet-b7`

#### ConvNeXt 系列
- `convnext-tiny`, `convnext-small`, `convnext-base`, `convnext-large`

#### ViT
- `vit` - Vision Transformer

### 文本编码器

#### CLIP 文本编码器（英文）
- `clip:ViT-B/32` - 默认推荐
- `clip:RN50` - ResNet-50 版本

#### BERT（中文）
- `bert-base-chinese` - 中文 BERT 模型

## 训练参数说明

### 基础参数

- `--data-dir`: 数据目录（按类别组织的文件夹）
- `--output-dir`: 输出目录（模型和日志保存位置）
- `--image-encoder`: 图像编码器名称
- `--text-encoder`: 文本编码器名称
- `--embed-dim`: 嵌入维度（默认 512）
- `--batch-size`: 批次大小（默认 16-32，根据 GPU 内存调整）
- `--epochs`: 训练轮数（默认 100-200）
- `--learning-rate`: 学习率（默认 1e-4）
- `--weight-decay`: 权重衰减（默认 0.01）
- `--temperature`: 温度参数（默认 0.07）
- `--img-size`: 图像大小（默认 224）
- `--augmentation`: 数据增强类型（`none`, `minimal`, `standard`）

### 交叉验证参数

- `--use-cv`: 启用 K 折交叉验证（**必须启用**）
- `--n-splits`: 交叉验证折数（默认 5）
- `--random-state`: 随机种子（默认 42）

### 类别不平衡处理

- `--use-weighted-sampling`: 启用加权采样
- `--weight-method`: 权重计算方法（`inverse_freq`, `inverse_sqrt`, `balanced`）
- `--weight-smooth-factor`: 权重平滑因子（默认 1.0）

### 其他参数

- `--num-workers`: 数据加载工作进程数（默认 4）
- `--gpu-id`: GPU ID（默认 0）
- `--no-amp`: 禁用混合精度训练（默认启用 AMP）
- `--class-texts-file`: 类别文本描述 JSON 文件路径
- `--text-template`: 文本模板（例如 "这是一张{class_name}的图片"）

## 数据准备

### 数据目录结构

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

### 类别文本描述文件（可选）

创建 `class_texts_hip_prosthesis.json`：

```json
{
  "类别1": "这是类别1的详细描述",
  "类别2": "这是类别2的详细描述",
  ...
}
```

## 训练输出

训练完成后，会在输出目录生成以下文件：

```
checkpoints/clip_models/resnet18_clip_ViT-B_32/
├── config.json                    # 训练配置
├── cv_summary.json                # 交叉验证汇总结果
├── fold_1/                        # 第1折
│   ├── checkpoint_best.pth        # 最佳模型
│   ├── checkpoint_latest.pth      # 最新模型
│   └── history.json               # 训练历史
├── fold_2/                        # 第2折
│   └── ...
└── ...
```

## 训练日志

日志文件保存在 `logs/clip_training/` 目录下：

```
logs/clip_training/
└── resnet18_clip_ViT-B_32_gpu6_20251205_150716.log
```

## 常见问题

### 1. CUDA Out of Memory (OOM)

**解决方案**：
- 减小 `BATCH_SIZE`（如从 32 改为 16 或 8）
- 减小 `NUM_WORKERS`（如从 4 改为 2）
- 使用更小的模型（如 `resnet18` 或 `starnet_s1`）

### 2. 训练速度慢

**解决方案**：
- 启用混合精度训练（`USE_AMP=true`）
- 增加 `NUM_WORKERS`
- 使用更小的模型或图像尺寸

### 3. 类别不平衡

**解决方案**：
- 启用加权采样（`--use-weighted-sampling`）
- 调整 `weight_method` 和 `weight_smooth_factor`

## 示例命令

### 单模型训练（ResNet18 + CLIP）

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/resnet18_clip \
    --image-encoder resnet18 \
    --text-encoder clip:ViT-B/32 \
    --batch-size 16 \
    --epochs 200 \
    --use-cv \
    --n-splits 5 \
    --gpu-id 6
```

### 使用 StarNet 预训练模型

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/starnet_s1_clip \
    --image-encoder starnet_s1:pretrained \
    --text-encoder clip:ViT-B/32 \
    --batch-size 16 \
    --epochs 200 \
    --use-cv \
    --n-splits 5 \
    --gpu-id 6
```

### 使用中文文本编码器

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/resnet18_bert \
    --image-encoder resnet18 \
    --text-encoder bert-base-chinese \
    --batch-size 16 \
    --epochs 200 \
    --use-cv \
    --n-splits 5 \
    --gpu-id 6
```

## 注意事项

1. **必须启用交叉验证**：当前版本只支持五折交叉验证训练
2. **数据目录结构**：确保数据按类别组织在子文件夹中
3. **GPU 内存**：根据 GPU 内存调整 `BATCH_SIZE`
4. **预训练权重**：StarNet 模型可以使用预训练权重，格式为 `starnet_s1:pretrained`
5. **文本编码器选择**：
   - 英文数据：使用 `clip:ViT-B/32`
   - 中文数据：使用 `bert-base-chinese`

## 训练监控

训练过程中可以查看日志文件：

```bash
tail -f logs/clip_training/resnet18_clip_ViT-B_32_gpu6_*.log
```

训练完成后，查看交叉验证汇总结果：

```bash
cat checkpoints/clip_models/resnet18_clip_ViT-B_32/cv_summary.json
```
