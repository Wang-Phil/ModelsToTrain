# SuperCLIP 快速开始指南

## 快速使用

### 1. 编辑配置文件

编辑 `train_clip_config.json`，确保包含 SuperCLIP 参数：

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
    "weight_smooth_factor": 1.0,
    "use_superclip_loss": true,
    "class_loss_weight": 1.0,
    "contrastive_loss_weight": 1.0
  }
]
```

### 2. 运行训练

```bash
bash run_train_clip.sh
```

就这么简单！脚本会自动：
- 从配置文件读取所有参数（包括 SuperCLIP 参数）
- 分配 GPU
- 启动训练
- 记录日志

### 3. 查看结果

训练完成后，结果保存在：
- 模型：`checkpoints/clip_models/{config_name}/fold_N/checkpoint_best.pth`
- 日志：`logs/clip_training/{config_name}_gpu{id}_{timestamp}.log`
- 汇总：`checkpoints/clip_models/{config_name}/cv_summary.json`

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `use_superclip_loss` | 启用 SuperCLIP 损失 | `false` |
| `class_loss_weight` | 分类损失权重 | `1.0` |
| `contrastive_loss_weight` | 对比损失权重 | `1.0` |

## 注意事项

1. **必须提供类别文本描述文件**：`class_texts_file` 是必需的，SuperCLIP 需要它来计算分类损失
2. **建议使用加权采样**：`use_weighted_sampling: true` 可以更好地处理类别不平衡
3. **交叉验证**：建议使用 `use_cv: true` 和 `n_splits: 5` 进行5折交叉验证

## 完整示例

查看 `SUPERCLIP_TRAINING_GUIDE.md` 获取更详细的说明。

