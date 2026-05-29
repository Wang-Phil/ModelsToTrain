#!/bin/bash

# 训练原始 BiomedCLIP 模型（仅使用对比损失）
# 不使用 CoOp，不使用 SCCM 损失，不使用分类损失

# 设置环境变量，优先使用本地缓存的模型
# 增加超时时间，避免网络连接问题
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_DOWNLOAD_MAX_RETRIES=10

# 检查模型是否已缓存
MODEL_CACHE_DIR="$HOME/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
if [ -d "${MODEL_CACHE_DIR}" ]; then
    echo "✓ 检测到模型已缓存，启用离线模式以避免网络问题"
    export HF_HUB_OFFLINE=1
else
    echo "警告: 模型未缓存，尝试使用镜像站点"
    # 如果无法连接 Hugging Face，尝试使用镜像站点
    if ! curl -s --connect-timeout 3 https://huggingface.co > /dev/null 2>&1; then
        echo "✓ 无法连接 Hugging Face，使用镜像站点: hf-mirror.com"
        export HF_ENDPOINT=https://hf-mirror.com
    fi
fi

# 数据目录
DATA_DIR="/home/ln/wangweicheng/ModelsTotrain/single_label_data"

# 输出目录
OUTPUT_DIR="output/biomedclip_original"

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
IMG_SIZE=224
AUGMENTATION="standard"
NUM_WORKERS=4
GPU_ID=2

# 交叉验证参数
N_SPLITS=5
RANDOM_STATE=42

# 早停参数
EARLY_STOPPING_PATIENCE=100
EARLY_STOPPING_MIN_DELTA=0.001
EARLY_STOPPING_MONITOR="val_loss"

# 加权采样（处理类别不平衡）
USE_WEIGHTED_SAMPLING=true
WEIGHT_METHOD="inverse_freq"
WEIGHT_SMOOTH_FACTOR=1.0

# 冻结图像编码器（可选，默认不冻结）
FREEZE_IMAGE_ENCODER=false

# 类别文本描述文件（可选，如果提供则会在训练时使用）
# CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"
CLASS_TEXTS_FILE=""

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 构建训练命令
CMD="python train_biomedcoop.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --img-size ${IMG_SIZE} \
    --augmentation ${AUGMENTATION} \
    --num-workers ${NUM_WORKERS} \
    --gpu-id ${GPU_ID} \
    --n-splits ${N_SPLITS} \
    --random-state ${RANDOM_STATE} \
    --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
    --early-stopping-min-delta ${EARLY_STOPPING_MIN_DELTA} \
    --early-stopping-monitor ${EARLY_STOPPING_MONITOR}"

# 添加类别文本描述文件（如果提供）
if [ -n "${CLASS_TEXTS_FILE}" ]; then
    CMD="${CMD} --class-texts-file ${CLASS_TEXTS_FILE}"
fi

# 添加加权采样
if [ "${USE_WEIGHTED_SAMPLING}" = true ]; then
    CMD="${CMD} --use-weighted-sampling --weight-method ${WEIGHT_METHOD} --weight-smooth-factor ${WEIGHT_SMOOTH_FACTOR}"
fi

# 添加冻结图像编码器
if [ "${FREEZE_IMAGE_ENCODER}" = true ]; then
    CMD="${CMD} --freeze-image-encoder"
fi

# 注意：由于模型已修改为原始 BiomedCLIP（不使用 CoOp），以下参数将被忽略：
# --n-ctx, --ctx-init, --csc, --class-token-position, --sccm-lambda, --use-focal-loss
# 这些参数在适配器中会被处理，但不会影响模型行为（因为 CoOp 已被注释）

# 打印配置信息
echo "============================================================"
echo "训练原始 BiomedCLIP 模型（仅使用对比损失）"
echo "============================================================"
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "批次大小: ${BATCH_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LEARNING_RATE}"
echo "交叉验证折数: ${N_SPLITS}"
echo "加权采样: ${USE_WEIGHTED_SAMPLING}"
echo "冻结图像编码器: ${FREEZE_IMAGE_ENCODER}"
echo "早停耐心值: ${EARLY_STOPPING_PATIENCE}"
echo ""
echo "注意："
echo "  - 模型使用原始 BiomedCLIP（不使用 CoOp）"
echo "  - 只使用对比损失（CLIP loss）"
echo "  - 不使用分类损失和 SCCM 损失"
echo "============================================================"
echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

# 执行训练
eval ${CMD}

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ 训练完成！"
    echo "============================================================"
    echo "输出目录: ${OUTPUT_DIR}"
    echo "可以查看以下文件："
    echo "  - ${OUTPUT_DIR}/cv_summary.json: 交叉验证结果汇总"
    echo "  - ${OUTPUT_DIR}/fold_*/best_model.pth: 每个 fold 的最佳模型"
    echo "  - ${OUTPUT_DIR}/fold_*/history.json: 每个 fold 的训练历史"
    echo "  - ${OUTPUT_DIR}/fold_*/train.log: 每个 fold 的训练日志（如果启用）"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "✗ 训练失败！"
    echo "============================================================"
    echo "请检查错误信息并修复问题"
    echo "============================================================"
    exit 1
fi
