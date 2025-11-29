# 多模型五折交叉验证并行训练指南

## 概述

本指南介绍如何使用脚本在不同GPU上并行运行多个模型的五折交叉验证。

## 脚本说明

### 1. `run_multi_model_cv.sh` - 并行版本（推荐）

**特点**:
- ✅ 多个模型同时在不同GPU上训练
- ✅ 自动分配GPU资源
- ✅ 自动等待GPU可用
- ✅ 后台运行，可以同时监控多个任务

**适用场景**: 有多个GPU，希望快速完成所有模型的训练

### 2. `run_multi_model_cv_sequential.sh` - 顺序版本（稳定）

**特点**:
- ✅ 按顺序训练模型，更稳定
- ✅ 每个模型训练完成后才开始下一个
- ✅ 实时输出训练日志
- ✅ 更容易调试问题

**适用场景**: 需要实时查看训练进度，或GPU资源有限

## 快速开始

### 步骤1: 配置脚本

编辑脚本文件，修改配置区域：

```bash
# 数据目录
DATA_DIR="single_label_data"

# GPU列表（根据您的系统修改）
GPUS=(0 1 2 3)

# 模型列表（要训练的模型）
MODELS=(
    "resnet18"
    "resnet50"
    "convnextv2_tiny"
)

# 训练参数
EPOCHS=50
BATCH_SIZE=32
LR=0.001
OPTIMIZER="adam"
LOSS="focal"
FOCAL_GAMMA=2.5
AUGMENTATION="strong"
```

### 步骤2: 运行脚本

#### 并行版本（推荐）

```bash
./run_multi_model_cv.sh
```

#### 顺序版本

```bash
./run_multi_model_cv_sequential.sh
```

## 配置说明

### GPU配置

```bash
# 单GPU
GPUS=(0)

# 多GPU
GPUS=(0 1 2 3)

# 指定特定GPU
GPUS=(1 3)  # 只使用GPU 1和3
```

### 模型配置

```bash
# 添加更多模型
MODELS=(
    "resnet18"
    "resnet50"
    "resnet101"
    "convnextv2_tiny"
    "convnextv2_base"
)
```

### 训练参数

所有训练参数都可以在脚本顶部配置：

```bash
EPOCHS=50                    # 训练轮数
BATCH_SIZE=32                # 批次大小
LR=0.001                     # 学习率
OPTIMIZER="adam"             # 优化器 (adam, adamw, sgd)
LOSS="focal"                 # 损失函数 (focal, ce, label_smoothing)
FOCAL_GAMMA=2.5              # Focal Loss的gamma参数
AUGMENTATION="strong"        # 数据增强 (standard, strong, medical)
WEIGHT_DECAY=0.001           # 权重衰减
EARLY_STOPPING_PATIENCE=10   # Early Stopping patience
USE_PRETRAINED=true          # 是否使用预训练权重
```

## 输出结构

训练完成后，输出结构如下：

```
checkpoints/cv_multi_models/
├── summary_report.txt          # 汇总报告
├── resnet18/
│   ├── cv_summary.json
│   ├── folds_info.json
│   └── fold_1/ ... fold_5/
├── resnet50/
│   ├── cv_summary.json
│   └── fold_1/ ... fold_5/
└── convnextv2_tiny/
    ├── cv_summary.json
    └── fold_1/ ... fold_5/

logs/cv_multi_models/
├── resnet18_gpu0.log
├── resnet50_gpu1.log
└── convnextv2_tiny_gpu2.log
```

## 监控训练

### 查看实时日志

```bash
# 查看特定模型的日志
tail -f logs/cv_multi_models/resnet18_gpu0.log

# 查看所有日志
tail -f logs/cv_multi_models/*.log
```

### 监控GPU使用情况

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 查看GPU使用率
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

### 查看训练进程

```bash
# 查看所有训练进程
ps aux | grep train_cross_validation

# 查看特定GPU的进程
nvidia-smi -i 0
```

## 汇总报告

训练完成后，会自动生成汇总报告：

```bash
cat checkpoints/cv_multi_models/summary_report.txt
```

报告包含：
- 各模型的平均mAP、Precision、Recall、F1
- 各模型的参数量和FLOPs
- 各模型的平均准确率

## 示例：完整训练流程

### 1. 准备数据

```bash
# 检查数据集
python prepare_cv_data.py --data-dir single_label_data --visualize
```

### 2. 配置脚本

编辑 `run_multi_model_cv.sh`，设置：
- GPU列表
- 模型列表
- 训练参数

### 3. 运行训练

```bash
# 并行版本
./run_multi_model_cv.sh

# 或顺序版本
./run_multi_model_cv_sequential.sh
```

### 4. 监控训练

```bash
# 在另一个终端查看日志
tail -f logs/cv_multi_models/*.log

# 监控GPU
watch -n 1 nvidia-smi
```

### 5. 查看结果

```bash
# 查看汇总报告
cat checkpoints/cv_multi_models/summary_report.txt

# 查看特定模型的结果
cat checkpoints/cv_multi_models/resnet18/cv_summary.json | python -m json.tool
```

## 高级用法

### 自定义GPU分配

如果需要手动指定每个模型使用的GPU，可以修改脚本中的分配逻辑：

```bash
# 在 run_cv_for_model 函数中
# 可以根据模型名称指定GPU
case $model in
    "resnet18")
        gpu_id=0
        ;;
    "resnet50")
        gpu_id=1
        ;;
    "convnextv2_tiny")
        gpu_id=2
        ;;
esac
```

### 添加更多模型

```bash
MODELS=(
    "resnet18"
    "resnet50"
    "resnet101"
    "convnextv2_tiny"
    "convnextv2_base"
    "efficientnetv2_s"
)
```

### 不同模型使用不同参数

如果需要为不同模型设置不同的参数，可以修改 `run_cv_for_model` 函数：

```bash
run_cv_for_model() {
    local model=$1
    local gpu_id=$2
    
    # 根据模型设置不同参数
    case $model in
        "resnet18")
            local epochs=30
            local lr=0.001
            ;;
        "resnet50")
            local epochs=50
            local lr=0.0005
            ;;
        *)
            local epochs=$EPOCHS
            local lr=$LR
            ;;
    esac
    
    # 使用这些参数构建命令
    ...
}
```

## 故障排除

### 问题1: GPU内存不足

**解决方案**:
- 减小批次大小: `BATCH_SIZE=16`
- 减少同时运行的模型数量
- 使用顺序版本脚本

### 问题2: 训练失败

**检查**:
1. 查看日志文件: `cat logs/cv_multi_models/<model>_gpu<id>.log`
2. 检查GPU状态: `nvidia-smi`
3. 检查数据路径是否正确

### 问题3: 进程卡住

**解决方案**:
```bash
# 查找并杀死卡住的进程
ps aux | grep train_cross_validation
kill -9 <PID>

# 清理GPU内存
nvidia-smi --gpu-reset -i <gpu_id>
```

## 性能优化建议

1. **GPU分配**: 根据模型大小合理分配GPU
   - 小模型（ResNet18）: 可以多个共享一个GPU
   - 大模型（ResNet50+）: 每个模型独占一个GPU

2. **批次大小**: 根据GPU内存调整
   - GPU内存 < 8GB: BATCH_SIZE=16
   - GPU内存 8-16GB: BATCH_SIZE=32
   - GPU内存 > 16GB: BATCH_SIZE=64

3. **训练顺序**: 先训练小模型，再训练大模型

## 总结

使用这些脚本可以：
- ✅ 充分利用多GPU资源
- ✅ 自动管理训练任务
- ✅ 生成统一的汇总报告
- ✅ 方便比较不同模型的性能

开始训练吧！🚀

