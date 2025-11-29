# 快速测试CLIP热力图生成

## 环境问题说明

如果遇到NumPy版本冲突，可以先修复环境：

```bash
# 降级numpy到兼容版本
pip install "numpy<2.0"

# 或者安装opencv-python
pip install opencv-python
```

## 快速测试步骤

### 1. 使用命令行工具测试

```bash
# 测试单张图片
python clip_gradcam_visualization.py \
    --model-path checkpoints/clip_models/resnet18_clip:ViT-B/32/fold_1/checkpoint_best.pth \
    --image-path data/test/Good\ Place/263_l.jpg \
    --class-texts-file class_texts_hip_prosthesis.json \
    --image-encoder resnet18 \
    --text-encoder clip:ViT-B/32 \
    --embed-dim 512 \
    --output-dir clip_gradcam_results
```

### 2. 使用Python脚本测试

运行测试脚本：
```bash
python test_clip_gradcam.py
```

### 3. 选择的测试图片

以下图片可用于测试：
- `data/test/Native Hip/346_r.jpg`
- `data/test/Good Place/263_l.jpg`  
- `data/test/Native Hip/838_l.jpg`
- `data/test/Acetabular Loosening/1012_r.jpg`
- `data/test/Fracture/60_l.jpg`

### 4. 类别文本

类别文本已保存在 `class_texts_hip_prosthesis.json`，包含9个类别：
- Good Place
- Native Hip
- Acetabular Loosening
- Stem Loosening
- Fracture
- Dislocation
- Infection
- Spacer
- Wear

## 输出结果

结果将保存在 `clip_gradcam_results/` 目录，包含：
- 完整可视化图（原始图像、热力图、叠加结果、预测概率）
- 叠加图像（仅原始图像与热力图的叠加）

## 如果遇到问题

1. **模型加载失败**：检查模型路径和配置是否正确
2. **依赖缺失**：安装缺失的包（opencv-python, matplotlib等）
3. **NumPy版本冲突**：降级numpy到1.x版本
4. **CUDA错误**：使用 `--cpu` 参数强制使用CPU

