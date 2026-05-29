#!/bin/bash

# ResNet50 (无预训练) + BiomedCLIP 文本编码器训练脚本
# 使用 ResNet50（从随机初始化）作为图像编码器，BiomedCLIP 的文本编码器进行训练

set -e

# 设置环境变量以避免警告
export TOKENIZERS_PARALLELISM=false

# ============================================
# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出目录（使用预训练权重）
OUTPUT_DIR="output/resnet50_pretrained_biomedclip"

# GPU ID
GPU_ID=8

# 训练参数
BATCH_SIZE=16
EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
EMBED_DIM=512
TEMPERATURE=0.07
IMG_SIZE=224
AUGMENTATION="standard"
NUM_WORKERS=4
USE_AMP=true

# 交叉验证参数
USE_CV=true
N_SPLITS=5
RANDOM_STATE=42

# ResNet50 模型（默认使用预训练权重）
# 使用 resnet50 表示使用预训练权重（ImageNet）
# 使用 resnet50:false 表示不使用预训练权重（从随机初始化开始）
IMAGE_ENCODER="resnet50"

# 文本编码器（BiomedCLIP）
TEXT_ENCODER="biomedclip_text"

# 类别文本描述文件（可选，如果存在会自动使用）
# 目前暂时不使用文本类别描述
# CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"

# 冻结文本编码器（只训练图像编码器）
FREEZE_TEXT_ENCODER=true

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

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

print_info "=========================================="
print_info "ResNet50 (预训练) + BiomedCLIP 训练配置"
print_info "=========================================="
print_info "数据目录: $DATA_DIR"
print_info "输出目录: $OUTPUT_DIR"
print_info "图像编码器: $IMAGE_ENCODER (使用 ImageNet 预训练权重)"
print_info "文本编码器: $TEXT_ENCODER"
print_info "GPU ID: $GPU_ID"
print_info "批次大小: $BATCH_SIZE"
print_info "训练轮数: $EPOCHS"
print_info "学习率: $LEARNING_RATE"
print_info "交叉验证: $USE_CV ($N_SPLITS 折)"
print_info "冻结文本编码器: $FREEZE_TEXT_ENCODER (只训练图像编码器)"
print_info "类别文本描述: 不使用（使用默认模板）"
print_info "=========================================="

# 构建训练命令
CMD="python train_clip.py"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --image-encoder $IMAGE_ENCODER"
CMD="$CMD --text-encoder $TEXT_ENCODER"
CMD="$CMD --embed-dim $EMBED_DIM"
CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --learning-rate $LEARNING_RATE"
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --img-size $IMG_SIZE"
CMD="$CMD --augmentation $AUGMENTATION"
CMD="$CMD --num-workers $NUM_WORKERS"
CMD="$CMD --gpu-id $GPU_ID"

# 交叉验证参数
if [ "$USE_CV" = true ]; then
    CMD="$CMD --use-cv"
    CMD="$CMD --n-splits $N_SPLITS"
    CMD="$CMD --random-state $RANDOM_STATE"
fi

# AMP 参数
if [ "$USE_AMP" = false ]; then
    CMD="$CMD --no-amp"
fi

# 类别文本描述文件（目前不使用）
# if [ -f "$CLASS_TEXTS_FILE" ]; then
#     CMD="$CMD --class-texts-file $CLASS_TEXTS_FILE"
# fi

# 冻结文本编码器
if [ "$FREEZE_TEXT_ENCODER" = true ]; then
    CMD="$CMD --freeze-text-encoder"
fi

# 执行训练
print_info "开始训练..."
print_info "执行命令: $CMD"
print_info ""

$CMD

if [ $? -eq 0 ]; then
    print_success "训练完成！"
    print_info "结果保存在: $OUTPUT_DIR"
else
    print_error "训练失败！"
    exit 1
fi

