#!/bin/bash

# 混合模型训练脚本（使用原始 CLIP ResNet50）
# 使用原始 CLIP 的 ResNet50 图像编码器 + BiomedCLIP 的文本编码器
# 只训练图像编码器，使用对比损失 + 分类损失

set -e

# 设置环境变量以避免警告
export TOKENIZERS_PARALLELISM=false

# ============================================
# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出目录
OUTPUT_DIR="output/hybrid_original_clip_resnet50_biomedclip"

# GPU ID
GPU_ID=2

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
IMG_SIZE=224
AUGMENTATION="standard"
NUM_WORKERS=4
USE_AMP=true

# 交叉验证参数
N_SPLITS=5
RANDOM_STATE=42

# 损失函数权重
CLASSIFICATION_LOSS_WEIGHT=0.33    # 分类损失权重
CONTRASTIVE_LOSS_WEIGHT=0.33       # 对比损失权重
DISTILLATION_LOSS_WEIGHT=0.34      # 蒸馏损失权重

# 类别文本描述文件（可选，如果存在会自动使用）
CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"

# 早停参数（可选）
EARLY_STOPPING_PATIENCE=200        # 早停耐心值
EARLY_STOPPING_MIN_DELTA=0.0      # 早停最小改进阈值
EARLY_STOPPING_MONITOR="val_mAP"  # 早停监控指标

# 加权采样参数（用于处理类别不平衡）
USE_WEIGHTED_SAMPLING=true       # 是否启用加权采样
WEIGHT_METHOD="inverse_freq"      # 权重计算方法
WEIGHT_SMOOTH_FACTOR=1.0          # 权重平滑因子

# ============================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    print_error "数据目录不存在: $DATA_DIR"
    exit 1
fi

# 检查 CLIP 库是否安装
print_info "检查 CLIP 库..."
python -c "import clip; print('✓ CLIP 库已安装')" 2>/dev/null || {
    print_error "CLIP 库未安装"
    print_info "请安装: pip install git+https://github.com/openai/CLIP.git"
    exit 1
}

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

print_info "=========================================="
print_info "混合模型训练配置（原始 CLIP ResNet50）"
print_info "=========================================="
print_info "数据目录: $DATA_DIR"
print_info "输出目录: $OUTPUT_DIR"
print_info ""
print_info "模型配置:"
print_info "  - 图像编码器: 原始 CLIP ResNet50 (训练, 1024维 -> 512维投影)"
print_info "  - 文本编码器: BiomedCLIP (冻结, 512维)"
print_info ""
print_info "训练参数:"
print_info "  - GPU ID: $GPU_ID"
print_info "  - 批次大小: $BATCH_SIZE"
print_info "  - 训练轮数: $EPOCHS"
print_info "  - 学习率: $LEARNING_RATE"
print_info "  - 权重衰减: $WEIGHT_DECAY"
print_info "  - 图像大小: $IMG_SIZE"
print_info "  - 数据增强: $AUGMENTATION"
print_info "  - 工作进程: $NUM_WORKERS"
print_info ""
print_info "损失函数:"
print_info "  - 分类损失权重: $CLASSIFICATION_LOSS_WEIGHT"
print_info "  - 对比损失权重: $CONTRASTIVE_LOSS_WEIGHT"
print_info "  - 蒸馏损失权重: $DISTILLATION_LOSS_WEIGHT"
print_info ""
print_info "交叉验证:"
print_info "  - 折数: $N_SPLITS"
print_info "  - 随机种子: $RANDOM_STATE"
print_info ""
print_info "早停:"
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
    print_info "  - 耐心值: $EARLY_STOPPING_PATIENCE epochs"
    print_info "  - 最小改进: $EARLY_STOPPING_MIN_DELTA"
    print_info "  - 监控指标: $EARLY_STOPPING_MONITOR"
else
    print_info "  - 未启用早停"
fi
print_info ""
if [ "$USE_WEIGHTED_SAMPLING" = true ]; then
    print_info "加权采样: 启用"
    print_info "  - 权重方法: $WEIGHT_METHOD"
    print_info "  - 平滑因子: $WEIGHT_SMOOTH_FACTOR"
else
    print_info "加权采样: 未启用"
fi
print_info ""
if [ -f "$CLASS_TEXTS_FILE" ]; then
    print_info "类别文本描述: $CLASS_TEXTS_FILE"
else
    print_info "类别文本描述: 使用默认模板"
fi
print_info "=========================================="

# 构建训练命令
CMD="python train_biomedcoop.py"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --model-type hybrid"
CMD="$CMD --use-original-clip-resnet50"  # 使用原始 CLIP ResNet50
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --img-size $IMG_SIZE"
CMD="$CMD --augmentation $AUGMENTATION"
CMD="$CMD --num-workers $NUM_WORKERS"
CMD="$CMD --gpu-id $GPU_ID"
CMD="$CMD --n-splits $N_SPLITS"
CMD="$CMD --random-state $RANDOM_STATE"

# AMP 参数
if [ "$USE_AMP" = false ]; then
    CMD="$CMD --no-amp"
fi

# 早停参数
if [ -n "$EARLY_STOPPING_PATIENCE" ]; then
    CMD="$CMD --early-stopping-patience $EARLY_STOPPING_PATIENCE"
    CMD="$CMD --early-stopping-min-delta $EARLY_STOPPING_MIN_DELTA"
    CMD="$CMD --early-stopping-monitor $EARLY_STOPPING_MONITOR"
fi

# 加权采样参数
if [ "$USE_WEIGHTED_SAMPLING" = true ]; then
    CMD="$CMD --use-weighted-sampling"
    CMD="$CMD --weight-method $WEIGHT_METHOD"
    CMD="$CMD --weight-smooth-factor $WEIGHT_SMOOTH_FACTOR"
fi

# 损失权重参数
CMD="$CMD --classification-loss-weight $CLASSIFICATION_LOSS_WEIGHT"
CMD="$CMD --contrastive-loss-weight $CONTRASTIVE_LOSS_WEIGHT"
CMD="$CMD --distillation-loss-weight $DISTILLATION_LOSS_WEIGHT"

# 类别文本描述文件
if [ -f "$CLASS_TEXTS_FILE" ]; then
    CMD="$CMD --class-texts-file $CLASS_TEXTS_FILE"
fi

# 执行训练
print_info "开始训练..."
print_info "执行命令: $CMD"
print_info ""

$CMD

if [ $? -eq 0 ]; then
    print_success "训练完成！"
    print_info "结果保存在: $OUTPUT_DIR"

    # 显示交叉验证结果
    if [ -f "$OUTPUT_DIR/cv_summary.json" ]; then
        print_info ""
        print_info "交叉验证结果摘要:"
        python -c "
import json
with open('$OUTPUT_DIR/cv_summary.json', 'r') as f:
    data = json.load(f)
print(f'平均最佳验证准确率: {data[\"average_best_val_acc\"]:.2f}% ± {data[\"std_best_val_acc\"]:.2f}%')
print(f'平均最佳验证mAP: {data[\"average_best_val_mAP\"]:.2f}% ± {data[\"std_best_val_mAP\"]:.2f}%')
print(f'平均最终验证准确率: {data[\"average_val_acc\"]:.2f}% ± {data[\"std_val_acc\"]:.2f}%')
print(f'平均最终验证mAP: {data[\"average_val_mAP\"]:.2f}% ± {data[\"std_val_mAP\"]:.2f}%')
        "
    fi
else
    print_error "训练失败！"
    exit 1
fi

