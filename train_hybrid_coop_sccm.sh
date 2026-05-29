#!/bin/bash
# ============================================================================
# 混合模型 + CoOp + SCCM 训练脚本
# PMC-CLIP ResNet50 图像编码器 + BiomedCLIP 文本编码器 + CoOp + SCCM 损失
# ============================================================================

cd /home/ln/wangweicheng/ModelsTotrain

# ============================================================================
# 配置参数（根据需要修改）
# ============================================================================

# 数据路径
DATA_DIR="single_label_data"
CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"

# 输出目录
OUTPUT_DIR="output/hybrid_coop_sccm_new_loss"

# 模型类型
MODEL_TYPE="hybrid_coop_sccm"

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=5e-5  # 降低学习率，更稳定
WEIGHT_DECAY=0.01

# CoOp 参数
N_CTX=4
CTX_INIT="a photo of a"

# 损失权重（加入对比损失，使用较小权重）
SCCM_LAMBDA=0
CLASSIFICATION_LOSS_WEIGHT=0.6  # 分类损失
CONTRASTIVE_LOSS_WEIGHT=0.1    # 对比损失（较小权重，因为原始值较大）
DISTILLATION_LOSS_WEIGHT=0.3   # 蒸馏损失

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
GPU_ID=1

# ============================================================================
# 运行训练
# ============================================================================

echo "=============================================="
echo "开始训练混合模型 + CoOp + SCCM"
echo "=============================================="
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型类型: ${MODEL_TYPE}"
echo "CoOp 参数: n_ctx=${N_CTX}, ctx_init=${CTX_INIT}"
echo "损失权重: SCCM=${SCCM_LAMBDA}, CE=${CLASSIFICATION_LOSS_WEIGHT}, "
echo "         Contrastive=${CONTRASTIVE_LOSS_WEIGHT}, Distill=${DISTILLATION_LOSS_WEIGHT}"
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
    --n-ctx ${N_CTX} \
    --ctx-init "${CTX_INIT}" \
    --sccm-lambda ${SCCM_LAMBDA} \
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

