#!/bin/bash

# LSAL (LLM-Semantic Adaptive Loss) BiomedCLIP 训练脚本
# 使用 Dassl 框架训练 LSAL_BiomedCLIP 模型

set -e

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6  # 设置使用的GPU ID

# ============================================
# 配置区域
# ============================================

# 数据目录（按类别组织的文件夹）
DATA_DIR="single_label_data"

# 输出目录
OUTPUT_DIR="output/lsal_biomedclip"

# 配置文件路径
CONFIG_FILE="configs/trainers/LSAL_BiomedCLIP/vit_b16.yaml"

# 语义文件目录（包含 class_centers.pt 和 soft_labels_matrix.pt）
SEMANTICS_DIR="semantics"

# 随机种子
SEED=42

# 训练参数
LEARNING_RATE=0.002
EPOCHS=100
BATCH_SIZE=32

# LSAL 特定参数
LAMBDA_ANCHOR=0.5  # Semantic Anchor Loss的权重系数

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

# 检查语义文件
print_info "检查语义文件..."
REQUIRED_FILES=(
    "$SEMANTICS_DIR/class_centers.pt"
    "$SEMANTICS_DIR/soft_labels_matrix.pt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "语义文件不存在: $file"
        print_info "请先运行: python models/build_llm_semantics.py --classnames_file class_texts_hip_prosthesis.json --templates_file hip_prosthesis_prompt_templates.py --output_dir $SEMANTICS_DIR"
        exit 1
    else
        print_info "✓ $(basename $file) 已存在"
    fi
done

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
print_info "输出目录: $OUTPUT_DIR"

# 创建配置文件目录（如果不存在）
mkdir -p "$(dirname "$CONFIG_FILE")"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    print_warning "配置文件不存在: $CONFIG_FILE"
    print_info "将使用默认配置和命令行参数"
fi

# 打印训练配置
print_info "训练配置:"
echo "  数据目录: $DATA_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo "  语义文件目录: $SEMANTICS_DIR"
echo "  随机种子: $SEED"
echo "  学习率: $LEARNING_RATE"
echo "  训练轮数: $EPOCHS"
echo "  批次大小: $BATCH_SIZE"
echo "  Lambda Anchor: $LAMBDA_ANCHOR"

echo ""
print_info "开始训练..."

# 运行训练脚本（使用Dassl框架）
# 确保使用当前目录的Dassl，而不是其他路径的
CURRENT_DIR=$(pwd)
# 清除可能干扰的PYTHONPATH，优先使用当前目录的Dassl
export PYTHONPATH="$CURRENT_DIR/Dassl.pytorch:$CURRENT_DIR/models:$CURRENT_DIR:$PYTHONPATH"

# 确保使用当前目录的Dassl
cd "$CURRENT_DIR/Dassl.pytorch"

# 使用绝对路径确保使用正确的配置和路径
python tools/train.py \
    --root "$CURRENT_DIR/$DATA_DIR" \
    --output-dir "$CURRENT_DIR/$OUTPUT_DIR" \
    --seed "$SEED" \
    --trainer "LSAL_BiomedCLIP" \
    --config-file "$CURRENT_DIR/$CONFIG_FILE" \
    OPTIM.LR "$LEARNING_RATE" \
    OPTIM.MAX_EPOCH "$EPOCHS" \
    DATASET.NUM_SHOTS -1 \
    TRAINER.LSAL.LAMBDA_ANCHOR "$LAMBDA_ANCHOR"

cd "$CURRENT_DIR"

if [ $? -eq 0 ]; then
    print_success "训练完成！"
    print_info "结果保存在: $OUTPUT_DIR"
else
    print_error "训练失败！"
    exit 1
fi

