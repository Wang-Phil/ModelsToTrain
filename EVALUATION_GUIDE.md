# 模型评估指南

## 概述

`evaluate_model.py` 是一个完整的模型性能评估脚本，支持多分类任务的全面评估，包括：

- **准确率指标**: 整体准确率、加权平均、宏平均、微平均
- **分类指标**: 精确率、召回率、F1分数、特异性
- **ROC分析**: ROC曲线、AUC值（微平均、宏平均、每类别）
- **混淆矩阵**: 可视化混淆矩阵
- **详细报告**: 每个类别的详细指标和分类报告

## 快速开始

### 基础评估

```bash
python evaluate_model.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/val \
    --output-dir evaluation_results
```

### 评估测试集

```bash
python evaluate_model.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/test \
    --output-dir evaluation_results/test_set
```

## 参数说明

### 必需参数

- `--model-path`: 模型检查点路径（.pth文件）
  - 支持从训练脚本保存的模型文件
  - 自动识别模型类型和类别信息

- `--data-dir`: 评估数据目录
  - 数据应按类别组织在子文件夹中
  - 例如: `data/val/Class1/`, `data/val/Class2/`, ...

### 可选参数

- `--num-classes`: 类别数
  - 如果模型文件中没有保存类别信息，需要手动指定
  - 默认: 从模型文件自动获取

- `--img-size`: 输入图像大小
  - 默认: 224
  - 应与训练时使用的图像大小一致

- `--batch-size`: 批次大小
  - 默认: 32
  - 根据GPU内存调整

- `--num-workers`: 数据加载线程数
  - 默认: 4
  - 建议设置为CPU核心数

- `--output-dir`: 输出目录
  - 默认: `evaluation_results`
  - 所有结果将保存到此目录

- `--cpu`: 强制使用CPU
  - 即使有GPU也使用CPU进行评估

## 输出文件

评估完成后，在输出目录中会生成以下文件：

### 1. `metrics.json`
包含所有评估指标的JSON文件：
```json
{
  "accuracy": 0.95,
  "precision_weighted": 0.94,
  "recall_weighted": 0.95,
  "f1_weighted": 0.94,
  "specificity_weighted": 0.98,
  "roc_auc_micro": 0.99,
  "roc_auc_macro": 0.98,
  "per_class_precision": [...],
  "per_class_recall": [...],
  "per_class_f1": [...],
  "per_class_specificity": [...],
  "per_class_auc": [...],
  "confusion_matrix": [[...], [...]],
  "classification_report": {...},
  "inference_time": {
    "avg_ms": 12.5,
    "total_sec": 45.2,
    "throughput": 22.1
  }
}
```

### 2. `classification_report.txt`
文本格式的分类报告，包含：
- 每个类别的精确率、召回率、F1分数
- 混淆矩阵

### 3. `confusion_matrix.png`
混淆矩阵可视化（百分比版本）

### 4. `confusion_matrix_count.png`
混淆矩阵可视化（原始数值版本）

### 5. `roc_curves.png`
每个类别的ROC曲线图

### 6. `per_class_metrics.png`
每个类别的指标对比图，包括：
- 精确率、召回率、F1分数柱状图
- 特异性柱状图
- AUC柱状图
- 混淆矩阵热力图

## 评估指标说明

### 整体指标

1. **准确率 (Accuracy)**
   - 正确预测的样本占总样本的比例
   - 公式: (TP + TN) / (TP + TN + FP + FN)

2. **加权平均 (Weighted Average)**
   - 根据每个类别的样本数量进行加权
   - 适用于类别不平衡的数据集

3. **宏平均 (Macro Average)**
   - 所有类别的指标平均值
   - 每个类别权重相等

4. **微平均 (Micro Average)**
   - 将所有类别的TP、FP、FN、TN相加后计算
   - 适用于多分类任务

### 分类指标

1. **精确率 (Precision)**
   - 预测为正类的样本中，实际为正类的比例
   - 公式: TP / (TP + FP)

2. **召回率 (Recall/Sensitivity)**
   - 实际为正类的样本中，被正确预测的比例
   - 公式: TP / (TP + FN)

3. **F1分数 (F1-Score)**
   - 精确率和召回率的调和平均
   - 公式: 2 * (Precision * Recall) / (Precision + Recall)

4. **特异性 (Specificity)**
   - 实际为负类的样本中，被正确预测为负类的比例
   - 公式: TN / (TN + FP)

### ROC分析

1. **ROC曲线 (ROC Curve)**
   - 以假正率(FPR)为横轴，真正率(TPR)为纵轴
   - 曲线下面积越大，模型性能越好

2. **AUC值 (Area Under Curve)**
   - ROC曲线下的面积
   - 范围: 0-1，1表示完美分类器
   - 微平均AUC: 将所有类别的预测合并后计算
   - 宏平均AUC: 每个类别AUC的平均值

## 使用示例

### 示例1: 评估验证集

```bash
python evaluate_model.py \
    --model-path checkpoints/resnet50/best_model.pth \
    --data-dir data/val \
    --output-dir evaluation_results/resnet50_val
```

### 示例2: 评估测试集（自定义参数）

```bash
python evaluate_model.py \
    --model-path checkpoints/convnextv2_tiny/best_model.pth \
    --data-dir data/test \
    --batch-size 16 \
    --img-size 256 \
    --num-workers 8 \
    --output-dir evaluation_results/convnextv2_test
```

### 示例3: CPU评估

```bash
python evaluate_model.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/val \
    --cpu \
    --batch-size 8 \
    --output-dir evaluation_results/cpu_eval
```

### 示例4: 评估特定模型

```bash
python evaluate_model.py \
    --model-path checkpoints/starnext_base/best_model.pth \
    --data-dir data/test \
    --num-classes 9 \
    --output-dir evaluation_results/starnext_test
```

## 结果解读

### 准确率解读

- **> 90%**: 优秀
- **80-90%**: 良好
- **70-80%**: 一般
- **< 70%**: 需要改进

### 混淆矩阵解读

- **对角线元素**: 正确分类的样本数
- **非对角线元素**: 错误分类的样本数
- **行**: 真实类别
- **列**: 预测类别

### ROC曲线解读

- **曲线越靠近左上角**: 模型性能越好
- **AUC = 1.0**: 完美分类器
- **AUC = 0.5**: 随机猜测
- **AUC < 0.5**: 模型性能差于随机猜测

## 常见问题

### 问题1: 类别不匹配

**错误**: `类别与模型不匹配`

**解决**: 
- 确保评估数据集的类别与训练时一致
- 使用 `--num-classes` 参数手动指定类别数

### 问题2: 内存不足

**错误**: `CUDA out of memory`

**解决**:
- 减小 `--batch-size`
- 使用 `--cpu` 参数在CPU上评估

### 问题3: 模型加载失败

**错误**: `无法加载模型`

**解决**:
- 检查模型路径是否正确
- 确保模型文件完整
- 检查模型架构是否匹配

### 问题4: 数据加载失败

**错误**: `无法加载数据`

**解决**:
- 检查数据目录结构是否正确
- 确保图像文件格式正确（.jpg, .png等）
- 检查文件权限

## 性能优化建议

1. **批次大小**: 
   - GPU: 32-64
   - CPU: 8-16

2. **数据加载**:
   - 设置合适的 `--num-workers`
   - 使用 `pin_memory=True`（GPU）

3. **推理速度**:
   - 使用GPU加速
   - 减小图像大小（如果允许）
   - 使用混合精度推理

## 批量评估

如果需要评估多个模型，可以创建脚本：

```python
import subprocess
import os

models = [
    'checkpoints/resnet50/best_model.pth',
    'checkpoints/convnextv2_tiny/best_model.pth',
    'checkpoints/starnext_base/best_model.pth',
]

for model_path in models:
    model_name = os.path.basename(os.path.dirname(model_path))
    output_dir = f'evaluation_results/{model_name}'
    
    cmd = [
        'python', 'evaluate_model.py',
        '--model-path', model_path,
        '--data-dir', 'data/test',
        '--output-dir', output_dir
    ]
    
    subprocess.run(cmd)
```

## 与训练脚本集成

评估脚本与训练脚本完全兼容：

1. 使用训练脚本保存的模型文件
2. 自动识别模型类型和类别信息
3. 使用相同的数据加载方式

## 联系与支持

如有问题，请检查：
1. 模型文件是否完整
2. 数据目录结构是否正确
3. 依赖包是否安装完整（特别是matplotlib和seaborn）

