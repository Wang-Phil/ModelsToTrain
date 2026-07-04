#!/bin/bash

# 知识蒸馏训练脚本
# 使用 BiomedCLIP 作为教师模型，蒸馏到 PMC-CLIP（学生模型）
# 学生模型使用 PMC-CLIP 的 ResNet50 图像编码器 + BiomedBERT 文本编码器

set -e

# 设置环境变量以避免警告
export TOKENIZERS_PARALLELISM=false

# ============================================
# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出目录
OUTPUT_DIR="output/distill_biomedclip_to_pmcclip"

# GPU ID
GPU_ID=0

# 训练参数
BATCH_SIZE=32
EPOCHS=200
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

# 学生模型配置
# 图像编码器: PMC-CLIP 的 ResNet50（使用 PMC-CLIP 预训练权重）
# 文本编码器: PMC-CLIP 的 BiomedBERT（使用 PMC-CLIP 预训练权重）
STUDENT_IMAGE_ENCODER="resnet50:pmcclip"
STUDENT_TEXT_ENCODER="pmcclip_text"

# 知识蒸馏参数
DISTILL_TEMPERATURE=4.0  # 蒸馏温度参数（用于软化概率分布）
DISTILL_ALPHA=0.5        # 蒸馏损失权重
CONTRASTIVE_WEIGHT=1.0   # CLIP对比损失权重
DISTILL_CLASS_LOSS_WEIGHT=1.0    # 分类损失权重（Focal Loss）
DISTILL_USE_FOCAL_LOSS=true      # 是否使用Focal Loss
DISTILL_FOCAL_ALPHA=0.25         # Focal Loss 的 alpha 参数
DISTILL_FOCAL_GAMMA=2.0          # Focal Loss 的 gamma 参数

# 类别文本描述文件（可选，如果存在会自动使用）
CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"

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
print_info "知识蒸馏训练配置 (BiomedCLIP -> PMC-CLIP)"
print_info "=========================================="
print_info "数据目录: $DATA_DIR"
print_info "输出目录: $OUTPUT_DIR"
print_info ""
print_info "教师模型: BiomedCLIP (完整模型，冻结)"
print_info "  - 图像编码器: ViT-B/16"
print_info "  - 文本编码器: PubMedBERT"
print_info ""
print_info "学生模型: PMC-CLIP"
print_info "  - 图像编码器: $STUDENT_IMAGE_ENCODER (PMC-CLIP预训练)"
print_info "  - 文本编码器: $STUDENT_TEXT_ENCODER (PMC-CLIP预训练)"
print_info ""
print_info "GPU ID: $GPU_ID"
print_info "批次大小: $BATCH_SIZE"
print_info "训练轮数: $EPOCHS"
print_info "学习率: $LEARNING_RATE"
print_info "交叉验证: $USE_CV ($N_SPLITS 折)"
print_info ""
print_info "蒸馏参数:"
print_info "  - 蒸馏温度: $DISTILL_TEMPERATURE"
print_info "  - 蒸馏损失权重: $DISTILL_ALPHA"
print_info "  - 对比损失权重: $CONTRASTIVE_WEIGHT"
print_info "  - 分类损失权重: $DISTILL_CLASS_LOSS_WEIGHT"
print_info "  - Focal Loss: $DISTILL_USE_FOCAL_LOSS (alpha=$DISTILL_FOCAL_ALPHA, gamma=$DISTILL_FOCAL_GAMMA)"
if [ -f "$CLASS_TEXTS_FILE" ]; then
    print_info "  - 类别文本描述: $CLASS_TEXTS_FILE"
else
    print_info "  - 类别文本描述: 使用默认模板"
fi
print_info "=========================================="

# 构建训练命令
CMD="python train_clip.py"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --image-encoder $STUDENT_IMAGE_ENCODER"
CMD="$CMD --text-encoder $STUDENT_TEXT_ENCODER"
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

# 知识蒸馏参数
CMD="$CMD --use-distillation"
CMD="$CMD --distill-temperature $DISTILL_TEMPERATURE"
CMD="$CMD --distill-alpha $DISTILL_ALPHA"
CMD="$CMD --contrastive-weight $CONTRASTIVE_WEIGHT"
CMD="$CMD --distill-class-loss-weight $DISTILL_CLASS_LOSS_WEIGHT"
if [ "$DISTILL_USE_FOCAL_LOSS" = false ]; then
    CMD="$CMD --distill-no-focal-loss"
fi
CMD="$CMD --distill-focal-alpha $DISTILL_FOCAL_ALPHA"
CMD="$CMD --distill-focal-gamma $DISTILL_FOCAL_GAMMA"

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
else
    print_error "训练失败！"
    exit 1
fi

