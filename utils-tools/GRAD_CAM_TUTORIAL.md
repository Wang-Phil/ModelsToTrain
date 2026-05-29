# Grad-CAM 可视化工具使用教程

## 简介

`grad_cam_starnet.py` 是一个用于生成 StarNet 模型 Grad-CAM 热力图的工具。它可以可视化模型在预测时关注的图像区域，帮助理解模型的决策过程。

## 功能特点

- ✅ 支持所有 StarNet 系列模型
- ✅ 自动识别目标层（推荐使用 `--auto-target-layer`）
- ✅ 支持自定义目标类别
- ✅ 多种颜色映射选项
- ✅ 生成原图、热力图和叠加图

## 安装依赖

确保已安装以下依赖：

```bash
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install numpy
pip install pillow
```

## 基本使用方法

### 1. 最简单的使用（自动选择目标层）

```bash
cd /home/ln/wangweicheng/ModelsTotrain
python utils-tools/grad_cam_starnet.py \
    --image-path path/to/your/image.jpg \
    --model starnet_s1_final \
    --checkpoint checkpoints/final_starnet_models/starnet_s1_final/fold_1/best_model.pth \
    --auto-target-layer
```

### 2. 指定目标类别

如果你想查看模型对特定类别的关注区域：

```bash
python utils-tools/grad_cam_starnet.py \
    --image-path path/to/your/image.jpg \
    --model starnet_s2_final \
    --checkpoint checkpoints/final_starnet_models/starnet_s2_final/fold_1/best_model.pth \
    --target-class 0 \
    --auto-target-layer
```

### 3. 自定义输出目录和参数

```bash
python utils-tools/grad_cam_starnet.py \
    --image-path path/to/your/image.jpg \
    --model starnet_s1_cross_with_gln \
    --checkpoint checkpoints/final_starnet_models/starnet_s1_cross_with_gln/fold_1/best_model.pth \
    --output-dir gradcam_results \
    --alpha 0.5 \
    --colormap jet \
    --auto-target-layer
```

## 参数说明

### 必需参数

- `--image-path`: 输入图像路径（支持 jpg, png 等格式）
- `--model`: 模型名称（例如：`starnet_s1_final`, `starnet_s2_final`, `starnet_s1_cross_with_gln`）
- `--checkpoint`: 模型检查点路径（.pth 文件）

### 可选参数

- `--output-dir`: 输出目录（默认：`gradcam_output`）
- `--target-class`: 目标类别索引（默认：使用模型预测的类别）
- `--num-classes`: 类别数量（默认：9）
- `--device`: 设备（默认：`cuda:0`，如果 CUDA 不可用会自动切换到 CPU）
- `--alpha`: 热力图透明度，范围 0-1（默认：0.4）
- `--colormap`: 颜色映射（可选：`jet`, `hot`, `viridis`, `plasma`，默认：`jet`）
- `--auto-target-layer`: **推荐使用**，自动选择目标层
- `--target-layer-name`: 手动指定目标层名称（例如：`stages.3.-1.dwconv2`）

## 支持的模型

所有在 `train_multiclass.py` 中注册的 StarNet 模型都支持，包括：

- `starnet_s1`, `starnet_s2`, `starnet_s3`, `starnet_s4`
- `starnet_s1_final`, `starnet_s2_final`, `starnet_s3_final`
- `starnet_s1_cross_with_gln`, `starnet_s2_cross_with_gln`
- `starnet_sa_s1`, `starnet_sa_s2`, `starnet_sa_s3`, `starnet_sa_s4`
- `starnet_s1_cross_star`, `starnet_s1_cross_star_add`, `starnet_s1_cross_star_samescale`
- 以及其他所有 StarNet 变体

## 使用示例

### 示例 1：查看模型对预测类别的关注区域

```bash
python utils-tools/grad_cam_starnet.py \
    --image-path single_label_data/Good\ Place/image_001.jpg \
    --model starnet_s1_final \
    --checkpoint checkpoints/final_starnet_models/starnet_s1_final/fold_1/best_model.pth \
    --auto-target-layer \
    --output-dir gradcam_output
```

### 示例 2：查看模型对特定类别的关注区域

假设你想查看模型在预测类别 3（"Fracture"）时的关注区域：

```bash
python utils-tools/grad_cam_starnet.py \
    --image-path single_label_data/Fracture/image_001.jpg \
    --model starnet_s2_final \
    --checkpoint checkpoints/final_starnet_models/starnet_s2_final/fold_1/best_model.pth \
    --target-class 3 \
    --auto-target-layer \
    --colormap hot
```

### 示例 3：批量处理（使用脚本）

创建一个脚本 `batch_gradcam.sh`：

```bash
#!/bin/bash

MODEL="starnet_s1_final"
CHECKPOINT="checkpoints/final_starnet_models/${MODEL}/fold_1/best_model.pth"
OUTPUT_DIR="gradcam_output/${MODEL}"

# 处理单个类别的所有图像
CLASS_DIR="single_label_data/Good Place"
for img in "$CLASS_DIR"/*.jpg; do
    python utils-tools/grad_cam_starnet.py \
        --image-path "$img" \
        --model "$MODEL" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --auto-target-layer
done
```

## 输出结果

工具会生成一个包含三张图的 PNG 文件：

1. **原图**：输入的原始图像
2. **热力图**：Grad-CAM 热力图，显示模型关注的区域（红色=高关注，蓝色=低关注）
3. **叠加图**：原图和热力图的叠加，更直观地显示关注区域

输出文件命名格式：`{图像名}_{模型名}_gradcam.png`

## 常见问题

### 1. 找不到目标层

**问题**：提示 "错误: 无法找到目标层"

**解决**：使用 `--auto-target-layer` 参数，或者手动指定层名称：

```bash
python utils-tools/grad_cam_starnet.py \
    ... \
    --target-layer-name "stages.3.-1.dwconv2"
```

### 2. CUDA 内存不足

**问题**：CUDA out of memory

**解决**：使用 CPU 模式：

```bash
python utils-tools/grad_cam_starnet.py \
    ... \
    --device cpu
```

### 3. 检查点格式不匹配

**问题**：模型加载失败

**解决**：检查点文件应该包含 `state_dict` 或 `model_state_dict` 键，或者直接是 state_dict。工具会自动处理 `module.` 前缀。

### 4. 图像加载失败

**问题**：无法加载图像

**解决**：确保图像路径正确，支持 jpg, png, bmp 等格式。

## 高级用法

### 查看不同层的 Grad-CAM

如果你想查看不同层的关注区域，可以手动指定层名称：

```bash
# 查看 Stage 2 的最后一个 block
python utils-tools/grad_cam_starnet.py \
    ... \
    --target-layer-name "stages.2.-1.dwconv2"

# 查看 Stage 3 的最后一个 block
python utils-tools/grad_cam_starnet.py \
    ... \
    --target-layer-name "stages.3.-1.dwconv2"
```

### 调整热力图透明度

```bash
# 更透明的热力图（alpha=0.3）
python utils-tools/grad_cam_starnet.py \
    ... \
    --alpha 0.3

# 更不透明的热力图（alpha=0.6）
python utils-tools/grad_cam_starnet.py \
    ... \
    --alpha 0.6
```

## 代码修复说明

已修复的问题：

1. ✅ 添加了缺失的 `transforms` 导入
2. ✅ 创建了 `StarNetGradCAMWrapper` 类，修复了 `get_loss` 方法，支持单张量输出（原 `grad_cam.py` 中的 `get_loss` 假设输出是元组）

## 注意事项

1. 确保模型检查点文件路径正确
2. 图像会被自动调整到 224x224 大小
3. 如果使用交叉验证训练的模型，需要指定对应的 fold 检查点
4. 建议使用 `--auto-target-layer` 参数，让工具自动选择最合适的层

## 输出示例

运行成功后，你会看到类似输出：

```
加载模型: starnet_s1_final
检查点: checkpoints/final_starnet_models/starnet_s1_final/fold_1/best_model.pth
✓ 模型加载成功
自动选择目标层...
✓ 选择的目标层: ['ConvBN(...)']
✓ 已保存可视化结果到: gradcam_output/image_001_starnet_s1_final_gradcam.png
  预测类别: 3, 置信度: 87.45%

✓ 完成！结果已保存到: gradcam_output/image_001_starnet_s1_final_gradcam.png
```

## 相关文件

- `grad_cam_starnet.py`: 主脚本
- `grad_cam.py`: Grad-CAM 核心实现
- `train_multiclass.py`: 模型创建函数

