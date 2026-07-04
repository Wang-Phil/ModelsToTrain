#!/bin/bash
# ============================================================================
# PMC-CLIP 完整模型训练脚本
# 使用 PMC-CLIP 图像编码器 + 文本编码器
# 支持分类损失、对比损失、蒸馏损失
# ============================================================================

# 进入项目目录
cd /home/ln/wangweicheng/ModelsTotrain

# ============================================================================
# 配置参数（根据需要修改）
# ============================================================================

# 数据路径
DATA_DIR="single_label_data"
CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"

# 输出目录
OUTPUT_DIR="output/pmcclip_full"

# 模型类型
MODEL_TYPE="pmcclip_full"

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01

# 损失权重
CLASSIFICATION_LOSS_WEIGHT=0.4
CONTRASTIVE_LOSS_WEIGHT=0.4
DISTILLATION_LOSS_WEIGHT=0.2

# 交叉验证
N_SPLITS=5
RANDOM_STATE=42

# 早停
EARLY_STOPPING_PATIENCE=100
EARLY_STOPPING_MONITOR="val_mAP"

# 加权采样
USE_WEIGHTED_SAMPLING="--use-weighted-sampling"
WEIGHT_METHOD="inverse_freq"

# GPU
GPU_ID=2

# ============================================================================
# 运行训练
# ============================================================================

echo "=============================================="
echo "开始训练 PMC-CLIP 完整模型"
echo "=============================================="
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型类型: ${MODEL_TYPE}"
echo "损失权重: CE=${CLASSIFICATION_LOSS_WEIGHT}, Contrastive=${CONTRASTIVE_LOSS_WEIGHT}, Distill=${DISTILLATION_LOSS_WEIGHT}"
echo "=============================================="

python train_biomedcoop.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-type "${MODEL_TYPE}" \
    --class-texts-file "${CLASS_TEXTS_FILE}" \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --classification-loss-weight ${CLASSIFICATION_LOSS_WEIGHT} \
    --contrastive-loss-weight ${CONTRASTIVE_LOSS_WEIGHT} \
    --distillation-loss-weight ${DISTILLATION_LOSS_WEIGHT} \
    ${USE_WEIGHTED_SAMPLING} \
    --weight-method ${WEIGHT_METHOD} \
    --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
    --early-stopping-monitor ${EARLY_STOPPING_MONITOR} \
    --n-splits ${N_SPLITS} \
    --random-state ${RANDOM_STATE} \
    --gpu-id ${GPU_ID}

echo "=============================================="
echo "训练完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "=============================================="

