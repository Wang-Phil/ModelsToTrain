# SuperCLIP 训练指南

本指南说明如何使用 `train_clip_config.json` 和 `run_train_clip.sh` 来训练 SuperCLIP 模型。

## 什么是 SuperCLIP？

SuperCLIP 是结合了分类损失和对比损失的 CLIP 训练方法：
- **分类损失**：使用加权交叉熵，处理类别不平衡问题
- **对比损失**：标准的 CLIP 对比学习损失
- **总损失**：`total_loss = class_loss_weight * class_loss + contrastive_loss_weight * contrastive_loss`

## 配置文件设置

### 1. 编辑 `train_clip_config.json`

在配置文件中添加 SuperCLIP 相关参数：

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
    "early_stopping_patience": null,
    "early_stopping_min_delta": 0.001,
    "early_stopping_monitor": "val_loss",
    "class_texts_file": "class_texts_hip_prosthesis.json",
    "use_weighted_sampling": true,
    "weight_method": "inverse_freq",
    "weight_smooth_factor": 1.0,
    
    // SuperCLIP 参数
    "use_superclip_loss": true,           // 启用 SuperCLIP 损失
    "class_loss_weight": 1.0,            // 分类损失权重
    "contrastive_loss_weight": 1.0        // 对比损失权重
  }
]
```

### 2. SuperCLIP 参数说明

- **`use_superclip_loss`** (boolean): 是否使用 SuperCLIP 损失函数
  - `true`: 使用 SuperCLIP 损失（分类损失 + 对比损失）
  - `false`: 使用标准 CLIP 损失（仅对比损失）

- **`class_loss_weight`** (float): 分类损失权重，默认 1.0
  - 增大此值会增加分类损失的权重
  - 建议范围：0.5 - 2.0

- **`contrastive_loss_weight`** (float): 对比损失权重，默认 1.0
  - 增大此值会增加对比损失的权重
  - 建议范围：0.5 - 2.0

## 使用方法

### 方法 1: 使用配置文件（推荐）

1. **编辑配置文件** `train_clip_config.json`，添加 SuperCLIP 参数

2. **运行训练脚本**：
```bash
bash run_train_clip.sh
```

脚本会自动：
- 从 `train_clip_config.json` 读取所有配置（包括 SuperCLIP 参数）
- 为每个配置分配 GPU
- 启动训练并记录日志

### 方法 2: 直接使用 Python 脚本

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/superclip_test \
    --config-file train_clip_config.json \
    --multi-config \
    --gpu-id 0
```

### 方法 3: 命令行参数（不使用配置文件）

```bash
python train_clip.py \
    --data-dir single_label_data \
    --output-dir checkpoints/clip_models/superclip_test \
    --image-encoder resnet18 \
    --text-encoder clip:ViT-B/32 \
    --use-cv \
    --n-splits 5 \
    --use-superclip-loss \
    --class-loss-weight 1.0 \
    --contrastive-loss-weight 1.0 \
    --class-texts-file class_texts_hip_prosthesis.json \
    --use-weighted-sampling \
    --gpu-id 0
```

## 训练脚本配置

### 编辑 `run_train_clip.sh`

确保以下设置正确：

```bash
# 数据目录
DATA_DIR="single_label_data"

# 输出基础目录
OUTPUT_BASE_DIR="checkpoints/clip_models"

# 配置文件路径
CONFIG_FILE="train_clip_config.json"

# 是否使用多配置训练
USE_MULTI_CONFIG=true

# GPU 列表
GPUS=(6)  # 根据实际情况修改
```

## 训练输出

训练完成后，会在输出目录生成：

```
checkpoints/clip_models/resnet18_clip_ViT-B_32/
├── config.json                    # 训练配置
├── cv_summary.json                # 交叉验证汇总结果
├── folds_info.json                # 各折的数据划分信息
├── fold_1/
│   ├── checkpoint_best.pth       # 最佳模型
│   ├── checkpoint_latest.pth     # 最新模型
│   └── history.json              # 训练历史
├── fold_2/
│   └── ...
└── ...
```

## 训练日志

日志文件保存在 `logs/clip_training/` 目录下，文件名格式：
```
{config_name}_gpu{gpu_id}_{timestamp}.log
```

## 监控训练

### 查看实时日志

```bash
tail -f logs/clip_training/resnet18_clip_ViT-B_32_gpu6_*.log
```

### 训练输出示例

使用 SuperCLIP 损失时，训练输出会显示分类损失和对比损失：

```
Epoch 1 [Train]: 100%|██████████| 202/202 [00:18<00:00, 11.26it/s, loss=2.6845, cls=1.5028, clip=1.1816, acc=12.25%]
[Val]: 100%|██████████| 202/202 [00:12<00:00, 16.18it/s, loss=2.6845, cls=1.5028, clip=1.1816, acc=14.11%]

Train Loss: 2.6845 (Class: 1.5028, Contrastive: 1.1816), Train Acc: 12.25%
Val Loss: 2.6845 (Class: 1.5028, Contrastive: 1.1816), Val Acc: 14.11%, Val mAP: 1.57%
```

## 参数调优建议

### 1. 损失权重平衡

- **默认配置**：`class_loss_weight=1.0, contrastive_loss_weight=1.0`
- **更重视分类**：`class_loss_weight=1.5, contrastive_loss_weight=0.8`
- **更重视对比学习**：`class_loss_weight=0.8, contrastive_loss_weight=1.5`

### 2. 与加权采样结合

SuperCLIP 损失可以与加权采样结合使用，更好地处理类别不平衡：

```json
{
  "use_superclip_loss": true,
  "use_weighted_sampling": true,
  "weight_method": "inverse_freq",
  "weight_smooth_factor": 1.0
}
```

### 3. 类别文本描述

确保提供类别文本描述文件（`class_texts_file`），SuperCLIP 需要它来计算分类损失。

## 常见问题

### Q: 训练时显示 "CLIPLoss.forward() got an unexpected keyword argument 'class_logits'"

**A**: 确保 `use_superclip_loss=true` 时，损失函数是 `SuperCLIPLoss` 而不是 `CLIPLoss`。检查配置文件中 `use_superclip_loss` 是否正确设置。

### Q: 如何只使用标准 CLIP 损失？

**A**: 在配置文件中设置 `"use_superclip_loss": false`，或删除该参数（默认为 false）。

### Q: 分类损失和对比损失的权重如何选择？

**A**: 建议从默认值（1.0, 1.0）开始，根据验证集表现调整。如果分类准确率低，可以增大 `class_loss_weight`；如果模型过拟合，可以增大 `contrastive_loss_weight`。

## 示例配置

### 完整示例：SuperCLIP + 加权采样 + 5折交叉验证

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
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "early_stopping_monitor": "val_loss",
    "class_texts_file": "class_texts_hip_prosthesis.json",
    "use_weighted_sampling": true,
    "weight_method": "inverse_freq",
    "weight_smooth_factor": 1.0,
    "use_superclip_loss": true,
    "class_loss_weight": 1.0,
    "contrastive_loss_weight": 1.0
  }
]
```

## 总结

1. 在 `train_clip_config.json` 中添加 SuperCLIP 参数
2. 运行 `bash run_train_clip.sh`
3. 监控训练日志，根据验证集表现调整损失权重
4. 训练完成后，在输出目录查看结果

祝训练顺利！

