#!/bin/bash

# BiomedCoOp 对比损失训练脚本
# 只训练图像编码器，只使用对比损失

# 参数设置
DATA=$1                    # 数据根目录
DATASET=$2                 # 数据集名称（如 singlelabel）
MODEL=$3                   # 模型名称（BiomedCLIP, CLIP, PubMedCLIP, PMCCLIP）
OUTPUT_DIR=$4              # 输出目录（可选）

# 默认参数
NCTX=4
CSC=False
CTP=end
N_SPLITS=5
RANDOM_STATE=42

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}

# 设置输出目录
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR=output/${DATASET}/cv_training/${TRAINER}_contrastive/nctx${NCTX}_csc${CSC}_ctp${CTP}
fi

echo "============================================================"
echo "BiomedCoOp 对比损失训练（只训练图像编码器）"
echo "============================================================"
echo "数据目录: ${DATA}"
echo "数据集: ${DATASET}"
echo "训练器: ${TRAINER}"
echo "输出目录: ${OUTPUT_DIR}"
echo "模型: ${MODEL}"
echo "============================================================"

# 运行交叉验证训练
python train_biomedcoop_cv.py \
    --root ${DATA} \
    --dataset ${DATASET} \
    --trainer ${TRAINER} \
    --output-dir ${OUTPUT_DIR} \
    --n-splits ${N_SPLITS} \
    --random-state ${RANDOM_STATE} \
    --seed ${RANDOM_STATE}

echo "============================================================"
echo "训练完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo "============================================================"

