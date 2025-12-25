# 按类别生成GradCAM热力图指南

## 功能说明

`generate_gradcam_by_class.py` 脚本可以为每个类别生成指定数量的GradCAM热力图。

## 使用方法

### 基本使用

```bash
python generate_gradcam_by_class.py
```

### 配置参数

脚本中的主要配置参数（可在代码中修改）：

- `checkpoint_path`: 模型检查点路径
- `image_encoder`: 图像编码器名称（如 "resnet18"）
- `text_encoder`: 文本编码器名称（如 "clip:ViT-B/32"）
- `embed_dim`: 嵌入维度（默认512）
- `img_size`: 图像大小（默认224）
- `num_images_per_class`: 每个类别生成的图片数量（默认30）
- `data_dir`: 数据目录（默认 "data"）
- `output_dir`: 输出目录（默认 "clip_gradcam_results_by_class"）

## 输出结构

脚本会在 `clip_gradcam_results_by_class` 目录下为每个类别创建一个子目录：

```
clip_gradcam_results_by_class/
├── Acetabular Loosening/
│   ├── image1_gradcam.png
│   ├── image1_gradcam_overlay.png
│   ├── image2_gradcam.png
│   ├── image2_gradcam_overlay.png
│   └── ...
├── Dislocation/
│   └── ...
├── Fracture/
│   └── ...
└── ...
```

每个类别目录包含：
- `*_gradcam.png`: 完整的可视化图（包含原始图像、热力图、叠加图和预测概率）
- `*_gradcam_overlay.png`: 仅叠加了热力图的图像

## 注意事项

1. **图片选择**: 如果某个类别的图片数量少于30张，会使用所有可用图片
2. **随机选择**: 使用随机种子42确保可重复性
3. **GPU支持**: 如果可用，会自动使用GPU加速
4. **错误处理**: 如果某张图片处理失败，会跳过并继续处理其他图片

## 示例输出

脚本会显示处理进度：

```
类别数量: 9
类别列表: ['Acetabular Loosening', 'Dislocation', ...]

加载模型: checkpoints/clip_models/resnet18_clip_ViT-B_32/checkpoint_best.pth
使用GPU

处理类别: Acetabular Loosening
  类别文本: The acetabular cup shows signs of loosening...
  图片数量: 30
  输出目录: clip_gradcam_results_by_class/Acetabular Loosening

生成 Acetabular Loosening 热力图: 100%|████████| 30/30 [00:45<00:00]
✓ 成功生成 30/30 张热力图
```

## 故障排除

1. **找不到模型文件**: 检查 `checkpoint_path` 是否正确
2. **找不到类别文本文件**: 确保 `class_texts_hip_prosthesis.json` 存在
3. **找不到图片**: 检查 `data/test/` 目录下是否有对应的类别文件夹
4. **内存不足**: 如果GPU内存不足，可以分批处理或减少 `num_images_per_class`

