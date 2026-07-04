#!/bin/bash

# BiomedCoOp PMC-CLIP 训练脚本（直接执行）
# 使用 train_biomedcoop.py 训练模型

set -e

# 设置环境变量
export TOKENIZERS_PARALLELISM=false

# ============================================
# 配置区域（根据实际情况修改）
# ============================================

# 数据目录（按类别组织的文件夹）
DATA_DIR="single_label_data"

# 输出目录
OUTPUT_DIR="output/biomedcoop_pmcclip"

# GPU ID
GPU_ID=2

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.002
WEIGHT_DECAY=0.0001

# 交叉验证参数
N_SPLITS=5
RANDOM_STATE=42

# BiomedCoOp 特定参数
N_CTX=4  # 上下文token数量
CTX_INIT="a photo of a"  # 上下文初始化文本
SCCM_LAMBDA=1.0  # SCCM损失权重
KDSP_LAMBDA=1.0  # KDSP损失权重
TAU=1.0  # 用于选择prompt的阈值
N_PROMPTS=4  # 使用的prompt数量

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

# 检查 PMC-CLIP 模型文件
PMC_CHECKPOINT_DIR="clip/checkpoints"
REQUIRED_FILES=(
    "image_encoder(resnet50).pth"
    "text_encoder.pth"
    "text_projection_layer.pth"
)

print_info "检查 PMC-CLIP 模型文件..."
for file in "${REQUIRED_FILES[@]}"; do
    filepath="$PMC_CHECKPOINT_DIR/$file"
    if [ ! -f "$filepath" ]; then
        print_warning "$file 未找到，训练时会自动下载"
    else
        print_info "✓ $file 已存在"
    fi
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
print_info "输出目录: $OUTPUT_DIR"

# 打印训练配置
print_info "训练配置:"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  GPU ID: $GPU_ID"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  权重衰减: $WEIGHT_DECAY"
echo "  交叉验证折数: $N_SPLITS"
echo "  随机种子: $RANDOM_STATE"
echo "  上下文token数量: $N_CTX"
echo "  上下文初始化: $CTX_INIT"
echo "  SCCM损失权重: $SCCM_LAMBDA"
echo "  KDSP损失权重: $KDSP_LAMBDA"
echo "  TAU: $TAU"
echo "  Prompt数量: $N_PROMPTS"
echo ""
print_info "开始训练..."

# 执行训练命令
python train_biomedcoop.py \
    --model-type pmcclip \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --n-ctx "$N_CTX" \
    --ctx-init "$CTX_INIT" \
    --sccm-lambda "$SCCM_LAMBDA" \
    --kdsp-lambda "$KDSP_LAMBDA" \
    --tau "$TAU" \
    --n-prompts "$N_PROMPTS" \
    --n-splits "$N_SPLITS" \
    --random-state "$RANDOM_STATE" \
    --gpu-id "$GPU_ID"

if [ $? -eq 0 ]; then
    print_success "训练完成！"
    print_info "结果保存在: $OUTPUT_DIR"
else
    print_error "训练失败！"
    exit 1
fi

