#!/bin/bash

# BiomedCoOp PMC-CLIP 训练命令集合
# 推荐使用 train_biomedcoop.py（支持 ImageFolder 格式）

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

# 类别文本描述文件（可选，用于更准确的模板）
# 如果存在，KDSP损失会使用更准确的模板
# 如果不存在，会使用默认模板 "a photo of a {classname}."
# 注意：现在会自动使用 hip_prosthesis_prompt_templates.py 中的类别描述
CLASS_TEXTS_FILE=""  # 例如: "class_texts_hip_prosthesis.json"

# ============================================

echo "=========================================="
echo "BiomedCoOp PMC-CLIP 训练命令"
echo "=========================================="
echo ""
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "GPU ID: $GPU_ID"
echo ""

# ============================================
# 命令1: 基础训练（5折交叉验证）
# ============================================

echo "命令1: 基础训练（5折交叉验证）"
echo "----------------------------------------"
echo "python train_biomedcoop.py \\"
echo "    --model-type pmcclip \\"
echo "    --data-dir $DATA_DIR \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --batch-size $BATCH_SIZE \\"
echo "    --epochs $EPOCHS \\"
echo "    --learning-rate $LEARNING_RATE \\"
echo "    --weight-decay $WEIGHT_DECAY \\"
echo "    --n-ctx $N_CTX \\"
echo "    --ctx-init \"$CTX_INIT\" \\"
echo "    --sccm-lambda $SCCM_LAMBDA \\"
echo "    --kdsp-lambda $KDSP_LAMBDA \\"
echo "    --tau $TAU \\"
echo "    --n-prompts $N_PROMPTS \\"
echo "    --n-splits $N_SPLITS \\"
echo "    --random-state $RANDOM_STATE \\"
echo "    --gpu-id $GPU_ID"
echo ""

# ============================================
# 命令2: 使用类别文本描述文件（如果存在）
# ============================================

if [ -n "$CLASS_TEXTS_FILE" ] && [ -f "$CLASS_TEXTS_FILE" ]; then
    echo "命令2: 使用类别文本描述文件"
    echo "----------------------------------------"
    echo "python train_biomedcoop.py \\"
    echo "    --model-type pmcclip \\"
    echo "    --data-dir $DATA_DIR \\"
    echo "    --output-dir $OUTPUT_DIR \\"
    echo "    --class-texts-file $CLASS_TEXTS_FILE \\"
    echo "    --batch-size $BATCH_SIZE \\"
    echo "    --epochs $EPOCHS \\"
    echo "    --learning-rate $LEARNING_RATE \\"
    echo "    --weight-decay $WEIGHT_DECAY \\"
    echo "    --n-ctx $N_CTX \\"
    echo "    --ctx-init \"$CTX_INIT\" \\"
    echo "    --sccm-lambda $SCCM_LAMBDA \\"
    echo "    --kdsp-lambda $KDSP_LAMBDA \\"
    echo "    --tau $TAU \\"
    echo "    --n-prompts $N_PROMPTS \\"
    echo "    --n-splits $N_SPLITS \\"
    echo "    --random-state $RANDOM_STATE \\"
    echo "    --gpu-id $GPU_ID"
    echo ""
fi

# ============================================
# 命令3: 单折训练（用于快速测试）
# ============================================

echo "命令3: 单折训练（用于快速测试）"
echo "----------------------------------------"
echo "python train_biomedcoop.py \\"
echo "    --model-type pmcclip \\"
echo "    --data-dir $DATA_DIR \\"
echo "    --output-dir ${OUTPUT_DIR}_single_fold \\"
echo "    --batch-size $BATCH_SIZE \\"
echo "    --epochs 10 \\"
echo "    --learning-rate $LEARNING_RATE \\"
echo "    --n-ctx $N_CTX \\"
echo "    --sccm-lambda $SCCM_LAMBDA \\"
echo "    --kdsp-lambda $KDSP_LAMBDA \\"
echo "    --n-splits 1 \\"
echo "    --gpu-id $GPU_ID"
echo ""

# ============================================
# 命令4: 调整损失权重
# ============================================

echo "命令4: 调整损失权重（更强调KDSP损失）"
echo "----------------------------------------"
echo "python train_biomedcoop.py \\"
echo "    --model-type pmcclip \\"
echo "    --data-dir $DATA_DIR \\"
echo "    --output-dir ${OUTPUT_DIR}_high_kdsp \\"
echo "    --batch-size $BATCH_SIZE \\"
echo "    --epochs $EPOCHS \\"
echo "    --learning-rate $LEARNING_RATE \\"
echo "    --n-ctx $N_CTX \\"
echo "    --sccm-lambda 1.0 \\"
echo "    --kdsp-lambda 2.0 \\"
echo "    --n-splits $N_SPLITS \\"
echo "    --gpu-id $GPU_ID"
echo ""

# ============================================
# 关于KDSP损失和类别描述文件的说明
# ============================================

echo "=========================================="
echo "关于KDSP损失和类别描述文件"
echo "=========================================="
echo ""
echo "KDSP损失说明:"
echo "  - KDSP损失使用预编码的模板特征（fixed_embeddings）"
echo "  - 这些特征在模型初始化时就已经计算好了"
echo "  - 不需要额外的类别描述文件也能正常工作"
echo ""
echo "类别描述文件（可选）:"
echo "  - 如果提供类别描述文件，KDSP损失会使用更准确的模板"
echo "  - 格式: JSON文件，键为类别名，值为描述文本"
echo "  - 示例: class_texts_hip_prosthesis.json"
echo "  - 如果不提供，会使用默认模板: \"a photo of a {classname}.\""
echo ""
echo "BIOMEDCOOP_TEMPLATES（已配置）:"
echo "  - 已自动导入 hip_prosthesis_prompt_templates.py"
echo "  - 包含 9 个髋关节假体类别的详细描述模板"
echo "  - 类别: Acetabular Loosening, Dislocation, Fracture, Good Place,"
echo "          Infection, Native Hip, Spacer, Stem Loosening, Wear"
echo "  - 每个类别有 50 个不同的描述模板"
echo "  - 训练时会自动使用这些模板生成 KDSP 损失的 teacher 特征"
echo ""
echo "模板优先级:"
echo "  1. trainers/prompt_templates.py 中的 BIOMEDCOOP_TEMPLATES（如果存在）"
echo "  2. hip_prosthesis_prompt_templates.py 中的 HIP_PROSTHESIS_TEMPLATES（已配置）"
echo "  3. 默认模板: \"a photo of a {classname}.\""
echo ""
echo "总结:"
echo "  ✓ KDSP损失会自动使用髋关节假体类别描述模板"
echo "  ✓ 每个类别有 50 个不同的描述，提供丰富的语义信息"
echo "  ✓ 训练时会从这些模板中选择合适的描述生成 teacher 特征"
echo ""

