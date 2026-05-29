#!/bin/bash

# ============================================================================
# BiomedCLIP 训练脚本
# 启用加权采样和 Focal Loss
# ============================================================================

# 设置GPU
export CUDA_VISIBLE_DEVICES=8

# 数据路径
DATA_DIR="single_label_data"
OUTPUT_DIR="output/biomedclip_focal_weighted"

# 模型配置
IMAGE_ENCODER="biomedclip"
TEXT_ENCODER="biomedclip_text"
EMBED_DIM=512

# 训练参数
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=1e-4  # 只训练图像编码器，使用较大的学习率（如果全参数微调，建议改为 1e-5）
WEIGHT_DECAY=0.01
TEMPERATURE=0.07

# 交叉验证
USE_CV=true
N_SPLITS=5
RANDOM_STATE=42

# 加权采样（处理类别不平衡）
USE_WEIGHTED_SAMPLING=true
WEIGHT_METHOD="inverse_freq"  # inverse_freq, inverse_sqrt, balanced
WEIGHT_SMOOTH_FACTOR=1.0

# SuperCLIP Loss 和 Focal Loss
USE_SUPERCLIP_LOSS=true
USE_FOCAL_LOSS=true
CLASS_LOSS_WEIGHT=1.0
CONTRASTIVE_LOSS_WEIGHT=1.0
FOCAL_ALPHA=0.25
FOCAL_GAMMA=2.0

# 早停
EARLY_STOPPING_PATIENCE=100
EARLY_STOPPING_MIN_DELTA=0.001
EARLY_STOPPING_MONITOR="val_loss"  # val_loss 或 val_acc

# 冻结参数
FREEZE_TEXT_ENCODER=true  # 冻结文本编码器（只训练图像编码器）
FREEZE_IMAGE_ENCODER=false  # 冻结图像编码器（只训练文本编码器）

# 其他参数
IMG_SIZE=224
AUGMENTATION="standard"
NUM_WORKERS=8
USE_AMP=true

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 打印配置信息
echo "============================================================"
echo "BiomedCLIP 训练配置"
echo "============================================================"
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "图像编码器: ${IMAGE_ENCODER}"
echo "文本编码器: ${TEXT_ENCODER}"
echo "嵌入维度: ${EMBED_DIM}"
echo ""
echo "训练参数:"
echo "  批次大小: ${BATCH_SIZE}"
echo "  训练轮数: ${EPOCHS}"
echo "  学习率: ${LEARNING_RATE}"
echo "  权重衰减: ${WEIGHT_DECAY}"
echo "  温度参数: ${TEMPERATURE}"
echo ""
echo "交叉验证:"
echo "  使用CV: ${USE_CV}"
echo "  折数: ${N_SPLITS}"
echo "  随机种子: ${RANDOM_STATE}"
echo ""
echo "加权采样:"
echo "  启用: ${USE_WEIGHTED_SAMPLING}"
echo "  权重方法: ${WEIGHT_METHOD}"
echo "  平滑因子: ${WEIGHT_SMOOTH_FACTOR}"
echo ""
echo "损失函数:"
echo "  SuperCLIP Loss: ${USE_SUPERCLIP_LOSS}"
echo "  Focal Loss: ${USE_FOCAL_LOSS}"
echo "  分类损失权重: ${CLASS_LOSS_WEIGHT}"
echo "  对比损失权重: ${CONTRASTIVE_LOSS_WEIGHT}"
echo "  Focal Alpha: ${FOCAL_ALPHA}"
echo "  Focal Gamma: ${FOCAL_GAMMA}"
echo ""
echo "早停:"
echo "  耐心值: ${EARLY_STOPPING_PATIENCE}"
echo "  最小改进: ${EARLY_STOPPING_MIN_DELTA}"
echo "  监控指标: ${EARLY_STOPPING_MONITOR}"
echo ""
echo "冻结参数:"
echo "  冻结文本编码器: ${FREEZE_TEXT_ENCODER}"
echo "  冻结图像编码器: ${FREEZE_IMAGE_ENCODER}"
echo "============================================================"
echo ""

# 构建训练命令
CMD="python train_clip.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --image-encoder ${IMAGE_ENCODER} \
    --text-encoder ${TEXT_ENCODER} \
    --embed-dim ${EMBED_DIM} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --weight-decay ${WEIGHT_DECAY} \
    --temperature ${TEMPERATURE} \
    --img-size ${IMG_SIZE} \
    --augmentation ${AUGMENTATION} \
    --num-workers ${NUM_WORKERS} \
    --random-state ${RANDOM_STATE}"

# 添加交叉验证参数
if [ "${USE_CV}" = true ]; then
    CMD="${CMD} --use-cv --n-splits ${N_SPLITS}"
fi

# 添加加权采样参数
if [ "${USE_WEIGHTED_SAMPLING}" = true ]; then
    CMD="${CMD} --use-weighted-sampling --weight-method ${WEIGHT_METHOD} --weight-smooth-factor ${WEIGHT_SMOOTH_FACTOR}"
fi

# 添加 SuperCLIP Loss 和 Focal Loss 参数
if [ "${USE_SUPERCLIP_LOSS}" = true ]; then
    CMD="${CMD} --use-superclip-loss \
        --class-loss-weight ${CLASS_LOSS_WEIGHT} \
        --contrastive-loss-weight ${CONTRASTIVE_LOSS_WEIGHT}"
    
    if [ "${USE_FOCAL_LOSS}" = true ]; then
        CMD="${CMD} --use-focal-loss \
            --focal-alpha ${FOCAL_ALPHA} \
            --focal-gamma ${FOCAL_GAMMA}"
    fi
fi

# 添加早停参数
if [ -n "${EARLY_STOPPING_PATIENCE}" ]; then
    CMD="${CMD} --early-stopping-patience ${EARLY_STOPPING_PATIENCE} \
        --early-stopping-min-delta ${EARLY_STOPPING_MIN_DELTA} \
        --early-stopping-monitor ${EARLY_STOPPING_MONITOR}"
fi

# 添加冻结参数
if [ "${FREEZE_TEXT_ENCODER}" = true ]; then
    CMD="${CMD} --freeze-text-encoder"
fi

if [ "${FREEZE_IMAGE_ENCODER}" = true ]; then
    CMD="${CMD} --freeze-image-encoder"
fi

# 添加混合精度训练
if [ "${USE_AMP}" = true ]; then
    # 默认启用，不需要额外参数
    :
else
    CMD="${CMD} --no-amp"
fi

# 打印完整命令
echo "执行命令:"
echo "${CMD}"
echo ""
echo "开始训练..."
echo ""

# 执行训练
eval ${CMD}

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "训练完成！"
    echo "============================================================"
    echo "结果保存在: ${OUTPUT_DIR}"
    echo ""
    echo "查看结果:"
    echo "  ls -lh ${OUTPUT_DIR}/fold_*/checkpoint/"
    echo "  cat ${OUTPUT_DIR}/fold_*/train.log"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "训练失败！"
    echo "============================================================"
    echo "请检查错误信息"
    exit 1
fi

