#!/bin/bash

# 交叉验证模型 Grad-CAM 可视化运行脚本
# 用法: bash run_gradcam_for_cv.sh

set -e

# ============ 配置区域 ============

# 模型目录（包含cv_summary.json和fold_X目录）
MODEL_DIR="checkpoints/final_models/sk_size_ablation/starnet_s1_sk39"

# 完整数据集目录（所有类别图像的根目录）
DATA_DIR="single_label_data"

# 输出目录（可选，默认为模型目录下的gradcam_output）
OUTPUT_DIR=""

# 要处理的fold（all表示处理所有fold，或指定如1,2,3）
FOLDS="all"

# 每个类别生成的样本数（None表示全部，或指定数字如10）
NUM_SAMPLES=""

# 目标类别（可选，用逗号分隔，如0,1,2表示只处理前3个类别。None表示所有类别）
TARGET_CLASSES=""

# 设备
DEVICE="cuda:0"

# ============ 运行 ============

echo "=========================================="
echo "交叉验证模型 Grad-CAM 可视化"
echo "=========================================="
echo "模型目录: $MODEL_DIR"
echo "数据目录: $DATA_DIR"
echo "输出目录: ${OUTPUT_DIR:-$MODEL_DIR/gradcam_output}"
echo "Fold: $FOLDS"
echo "每个类别样本数: ${NUM_SAMPLES:-全部}"
echo "目标类别: ${TARGET_CLASSES:-全部}"
echo "设备: $DEVICE"
echo "=========================================="

# 构建命令
CMD="python generate_cv_gradcam.py"
CMD="$CMD --model-dir $MODEL_DIR"
CMD="$CMD --data-dir $DATA_DIR"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

CMD="$CMD --folds $FOLDS"

if [ -n "$NUM_SAMPLES" ]; then
    CMD="$CMD --num-samples $NUM_SAMPLES"
fi

if [ -n "$TARGET_CLASSES" ]; then
    CMD="$CMD --target-classes $TARGET_CLASSES"
fi

CMD="$CMD --device $DEVICE"

# 运行命令
echo "执行命令: $CMD"
echo ""
$CMD

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="

