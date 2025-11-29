# CLIP模型热力图生成指南

本指南介绍如何使用 `clip_gradcam_visualization.py` 为CLIP模型生成Grad-CAM热力图，可视化图像中哪些区域与特定文本描述最相关。

## 功能特点

- **图像-文本对齐可视化**：显示图像中哪些区域与特定文本描述最相关
- **自动目标层检测**：自动识别不同图像编码器（ResNet、StarNet、ConvNeXt等）的目标层
- **多类别支持**：支持同时可视化多个类别的文本描述
- **灵活配置**：支持从命令行或文件加载类别文本描述

## 安装依赖

确保已安装必要的依赖：
```bash
pip install torch torchvision pillow opencv-python matplotlib numpy tqdm
```

## 使用方法

### 基本用法

```bash
python clip_gradcam_visualization.py \
    --model-path checkpoints/clip_model.pth \
    --image-path data/test/image.jpg \
    --class-texts "正常" "异常" "病灶" \
    --output-dir clip_gradcam_results
```

### 从文件加载类别文本

如果类别文本描述保存在JSON文件中：

```bash
python clip_gradcam_visualization.py \
    --model-path checkpoints/clip_model.pth \
    --image-path data/test/image.jpg \
    --class-texts-file class_texts_hip_prosthesis.json \
    --output-dir clip_gradcam_results
```

JSON文件格式：
```json
{
    "类别1": "这是一个类别的详细描述",
    "类别2": "这是另一个类别的详细描述",
    ...
}
```

### 可视化特定类别

如果想可视化特定类别（而不是最相似的类别）的热力图：

```bash
python clip_gradcam_visualization.py \
    --model-path checkpoints/clip_model.pth \
    --image-path data/test/image.jpg \
    --class-texts "正常" "异常" "病灶" \
    --target-text-idx 1 \
    --output-dir clip_gradcam_results
```

`--target-text-idx 1` 表示可视化索引为1的类别（即"异常"）的热力图。

### 完整参数说明

```bash
python clip_gradcam_visualization.py \
    --model-path PATH              # 模型检查点路径（必需）
    --image-path PATH              # 图像路径（必需）
    --class-texts TEXT [TEXT ...]  # 类别文本描述列表（与--class-texts-file二选一）
    --class-texts-file PATH        # 类别文本描述JSON文件（与--class-texts二选一）
    --target-text-idx INT          # 目标文本索引（None表示使用最相似的）
    --image-encoder NAME           # 图像编码器名称（如果无法从checkpoint获取）
    --text-encoder NAME            # 文本编码器名称（默认: bert-base-chinese）
    --embed-dim INT                # 嵌入维度（默认: 512）
    --img-size INT                 # 图像大小（默认: 224）
    --output-dir PATH              # 输出目录（默认: clip_gradcam_results）
    --output-name NAME             # 输出文件名（可选，不指定则自动生成）
    --cpu                          # 强制使用CPU
```

## 输出结果

脚本会生成两种可视化结果：

1. **完整可视化图** (`clip_gradcam_*.png`)：
   - 原始图像
   - Grad-CAM热力图
   - 叠加结果
   - 预测概率分布

2. **叠加图像** (`clip_gradcam_*_overlay.png`)：
   - 仅包含原始图像与热力图的叠加结果

## 支持的图像编码器

脚本支持以下图像编码器，并会自动识别目标层：

- **ResNet系列**：resnet18, resnet34, resnet50, resnet101, resnet152
- **StarNet系列**：starnet_s1, starnet_s2, starnet_s3, starnet_s4, starnet_dual_pyramid_rcf
- **ConvNeXt系列**：convnext-tiny, convnext-small, convnext-base, convnext-large
- **EfficientNet系列**：efficientnet-b0 ~ efficientnet-b7
- **ViT**：vit

## 示例

### 示例1：可视化髋关节假体图像

```bash
python clip_gradcam_visualization.py \
    --model-path checkpoints/clip_starnet_bert.pth \
    --image-path data/test/hip_prosthesis/image_001.jpg \
    --class-texts-file class_texts_hip_prosthesis.json \
    --output-dir results/gradcam
```

### 示例2：可视化特定类别

```bash
python clip_gradcam_visualization.py \
    --model-path checkpoints/clip_model.pth \
    --image-path test_image.jpg \
    --class-texts "正常假体" "假体松动" "假体周围感染" \
    --target-text-idx 2 \
    --output-name result_with_infection.png
```

这将可视化"假体周围感染"类别的热力图。

## 工作原理

1. **加载模型和图像**：加载训练好的CLIP模型和待分析的图像
2. **编码文本**：使用文本编码器将所有类别文本描述编码为特征向量
3. **前向传播**：图像通过图像编码器，在目标层捕获激活值
4. **计算相似度**：计算图像特征与文本特征的相似度
5. **反向传播**：通过相似度损失反向传播，获取目标层的梯度
6. **生成热力图**：使用梯度加权激活值生成CAM热力图
7. **可视化**：将热力图叠加到原始图像上

## 注意事项

1. **目标层选择**：如果自动检测的目标层不正确，可以手动指定。需要修改代码中的 `get_target_layer_for_clip` 函数。

2. **内存使用**：对于大图像或大批量，可能需要调整 `img_size` 或使用CPU模式。

3. **文本编码器**：确保文本编码器与训练时使用的相同（默认是 `bert-base-chinese`）。

4. **模型结构**：如果使用自定义的CLIP模型结构，可能需要修改 `get_target_layer_for_clip` 函数以正确识别目标层。

## 常见问题

### Q: 热力图显示为全黑或全白？

A: 这可能是因为：
- 目标层选择不正确，尝试手动指定目标层
- 梯度消失，检查模型是否正确加载
- 图像与文本完全不相关，尝试使用不同的文本描述

### Q: 如何可视化多个图像？

A: 可以编写一个简单的循环脚本：

```python
import os
from pathlib import Path
import subprocess

image_dir = "data/test"
output_dir = "results/gradcam"
class_texts_file = "class_texts.json"

for image_path in Path(image_dir).glob("*.jpg"):
    subprocess.run([
        "python", "clip_gradcam_visualization.py",
        "--model-path", "checkpoints/clip_model.pth",
        "--image-path", str(image_path),
        "--class-texts-file", class_texts_file,
        "--output-dir", output_dir
    ])
```

### Q: 如何在Python代码中使用？

A: 可以直接导入函数：

```python
from clip_gradcam_visualization import visualize_clip_gradcam, CLIPGradCAM
from models.clip import CLIPModel

# 加载模型
model = CLIPModel(...)
model.load_state_dict(torch.load("checkpoint.pth"))

# 生成可视化
class_texts = ["类别1", "类别2", "类别3"]
visualize_clip_gradcam(
    model=model,
    image_path="test.jpg",
    class_texts=class_texts,
    save_path="result.png"
)
```

## 参考

- [Grad-CAM论文](https://arxiv.org/abs/1610.02391)
- [CLIP论文](https://arxiv.org/abs/2103.00020)

