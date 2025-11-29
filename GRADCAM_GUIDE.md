# Grad-CAM可视化指南

## 概述

`gradcam_visualization.py` 是一个通用的Grad-CAM可视化工具，用于生成模型关注热力图，帮助理解模型的决策过程。

## 功能特性

- ✅ **自动识别目标层**: 支持多种模型架构，自动找到合适的卷积层
- ✅ **单图像模式**: 对单张图像生成Grad-CAM可视化
- ✅ **批量模式**: 批量处理多张图像
- ✅ **灵活配置**: 支持指定目标类别或使用预测类别
- ✅ **高质量输出**: 生成包含原图、热力图和叠加图的对比可视化

## 快速开始

### 单张图像可视化

```bash
python gradcam_visualization.py \
    --model-path checkpoints/best_model.pth \
    --image-path data/test/Dislocation/1004_l.jpg \
    --output-dir gradcam_results
```

### 批量图像可视化

```bash
python gradcam_visualization.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/val \
    --output-dir gradcam_results/batch \
    --num-samples 50
```

## 参数说明

### 必需参数

- `--model-path`: 模型检查点路径（.pth文件）
  - 支持从训练脚本保存的模型文件
  - 自动识别模型类型

### 数据相关（二选一）

- `--image-path`: 单张图像路径（单图像模式）
- `--data-dir`: 数据目录（批量模式，按类别组织）

### 可选参数

- `--model-name`: 模型名称
  - 如果无法从checkpoint自动获取，需要手动指定
  - 例如: `resnet50`, `convnextv2_tiny`, `starnext_base`

- `--target-category`: 目标类别索引
  - 如果为None，使用模型预测的类别
  - 可以指定特定类别来查看模型对该类别的关注区域

- `--img-size`: 输入图像大小
  - 默认: 224
  - 应与训练时使用的图像大小一致

- `--output-dir`: 输出目录
  - 默认: `gradcam_results`
  - 所有结果将保存到此目录

- `--num-samples`: 处理的样本数（批量模式）
  - 如果为None，处理所有样本
  - 默认: None

- `--batch-size`: 批次大小（批量模式）
  - 默认: 8
  - 根据GPU内存调整

- `--num-workers`: 数据加载线程数
  - 默认: 4

- `--cpu`: 强制使用CPU
  - 即使有GPU也使用CPU

## 支持的模型架构

脚本自动识别以下模型的目标层：

### Classic Models
- **ResNet**: `layer4[-1]` (最后一个残差块的最后一个卷积层)
- **DenseNet**: `features[-1]` (最后一个特征层)
- **MobileNet**: `features[-1]` (最后一个特征层)
- **EfficientNet**: `blocks[-1]` 或 `features[-1]`
- **Inception**: `Mixed_7c`
- **GoogleNet**: `inception5b`

### 现代架构
- **ConvNeXtV2**: `stages[-1][-1]` (最后一个stage的最后一个block)
- **StarNeXt**: `stages[-1][-1]`
- **StarNet**: `stages[-1][-1]`
- **MogaNet**: `stages[-1][-1]`

如果无法自动识别，脚本会尝试查找最后一个卷积层。

## 输出文件

### 单图像模式

生成一个包含三张子图的PNG文件：
- **原图**: 原始输入图像
- **热力图**: Grad-CAM热力图（颜色映射）
- **叠加图**: 原图与热力图的叠加

同时生成单独的叠加图像（`_overlay.png`）。

### 批量模式

为每个样本生成一个可视化文件，命名格式：
```
gradcam_XXXX_true_ClassName_pred_PredName.png
```

其中：
- `XXXX`: 样本索引
- `ClassName`: 真实类别名称
- `PredName`: 预测类别名称

## 使用示例

### 示例1: 单张图像可视化

```bash
python gradcam_visualization.py \
    --model-path checkpoints/resnet50/best_model.pth \
    --image-path data/test/Dislocation/1004_l.jpg \
    --output-dir gradcam_results/single
```

### 示例2: 批量处理验证集

```bash
python gradcam_visualization.py \
    --model-path checkpoints/convnextv2_tiny/best_model.pth \
    --data-dir data/val \
    --output-dir gradcam_results/val_set \
    --num-samples 100 \
    --batch-size 16
```

### 示例3: 指定目标类别

查看模型对特定类别的关注区域：

```bash
python gradcam_visualization.py \
    --model-path checkpoints/best_model.pth \
    --image-path data/test/Dislocation/1004_l.jpg \
    --target-category 1 \
    --output-dir gradcam_results/specific
```

### 示例4: 测试集可视化

```bash
python gradcam_visualization.py \
    --model-path checkpoints/starnext_base/best_model.pth \
    --data-dir data/test \
    --output-dir gradcam_results/test_set \
    --num-samples 200
```

### 示例5: CPU模式

```bash
python gradcam_visualization.py \
    --model-path checkpoints/best_model.pth \
    --image-path data/test/Dislocation/1004_l.jpg \
    --cpu \
    --output-dir gradcam_results/cpu
```

## 结果解读

### 热力图颜色含义

- **红色/黄色区域**: 模型高度关注的区域（对预测贡献大）
- **蓝色区域**: 模型较少关注的区域
- **叠加图**: 显示模型关注区域与原始图像的对应关系

### 分析建议

1. **正确预测的样本**:
   - 检查模型是否关注了正确的区域
   - 热力图是否覆盖了关键特征

2. **错误预测的样本**:
   - 查看模型关注了哪些区域
   - 是否关注了无关区域或背景
   - 帮助理解模型的错误原因

3. **不同类别的对比**:
   - 对比不同类别样本的热力图
   - 了解模型如何区分不同类别

## 常见问题

### 问题1: 无法找到目标层

**错误**: `无法自动找到目标层`

**解决**:
- 检查模型架构是否正确
- 尝试手动指定模型名称: `--model-name resnet50`
- 如果仍然失败，可能需要修改代码手动指定目标层

### 问题2: 内存不足

**错误**: `CUDA out of memory`

**解决**:
- 减小 `--batch-size`（批量模式）
- 使用 `--cpu` 参数
- 减小 `--num-samples`

### 问题3: 热力图全黑或全白

**可能原因**:
- 目标层选择不当
- 模型未正确加载
- 梯度未正确传播

**解决**:
- 检查模型是否正确加载
- 尝试不同的目标层
- 确保模型处于eval模式

### 问题4: 图像尺寸不匹配

**错误**: 图像尺寸与模型输入不匹配

**解决**:
- 使用 `--img-size` 参数指定正确的图像大小
- 确保与训练时使用的图像大小一致

## 性能优化建议

1. **批量处理**:
   - 使用合适的 `--batch-size`（GPU: 8-16, CPU: 2-4）
   - 批量模式比单图像模式更高效

2. **样本数量**:
   - 使用 `--num-samples` 限制处理数量
   - 先处理少量样本测试效果

3. **GPU加速**:
   - 默认使用GPU（如果可用）
   - CPU模式较慢但更稳定

## 与评估脚本集成

Grad-CAM可视化可以与评估脚本配合使用：

1. **先评估模型性能**:
   ```bash
   python evaluate_model.py \
       --model-path checkpoints/best_model.pth \
       --data-dir data/test \
       --output-dir evaluation_results
   ```

2. **再生成Grad-CAM可视化**:
   ```bash
   python gradcam_visualization.py \
       --model-path checkpoints/best_model.pth \
       --data-dir data/test \
       --output-dir gradcam_results
   ```

3. **分析结果**:
   - 查看评估报告了解整体性能
   - 查看Grad-CAM可视化了解模型关注点
   - 结合两者分析模型的优缺点

## 高级用法

### 自定义目标层

如果需要使用特定的层，可以修改 `get_target_layers` 函数或直接在代码中指定：

```python
# 在gradcam_visualization.py中
target_layers = [model.layer3[-1]]  # 使用layer3而不是layer4
```

### 批量处理特定类别

可以修改代码，只处理特定类别的样本：

```python
# 在visualize_gradcam_batch函数中添加过滤
if label != target_class:
    continue
```

## 联系与支持

如有问题，请检查：
1. 模型文件是否完整
2. 图像路径是否正确
3. 依赖包是否安装完整（特别是opencv-python和matplotlib）

