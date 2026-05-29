#!/bin/bash

# 训练脚本：使用对比损失训练多种模型类型
# 支持：biomedclip, clip, pmcclip, pubmedclip

# 默认参数
DATA_DIR="/home/ln/wangweicheng/ModelsTotrain/single_label_data"
OUTPUT_BASE_DIR="output"
MODEL_TYPE="biomedclip"  # biomedclip, clip, pmcclip, pubmedclip
CLIP_BACKBONE="ViT-B/16"  # ViT-B/16, ViT-B/32, RN50, RN101
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
EARLY_STOPPING_MONITOR="val_mAP"

# 加权采样（处理类别不平衡）
USE_WEIGHTED_SAMPLING=true
WEIGHT_METHOD="inverse_freq"
WEIGHT_SMOOTH_FACTOR=1.0

# 冻结图像编码器（可选，默认不冻结）
FREEZE_IMAGE_ENCODER=false

# 类别文本描述文件（可选，仅 biomedclip 支持）
CLASS_TEXTS_FILE=""

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --clip-backbone)
            CLIP_BACKBONE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --n-splits)
            N_SPLITS="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --class-texts-file)
            CLASS_TEXTS_FILE="$2"
            shift 2
            ;;
        --no-weighted-sampling)
            USE_WEIGHTED_SAMPLING=false
            shift
            ;;
        --freeze-image-encoder)
            FREEZE_IMAGE_ENCODER=true
            shift
            ;;
        -h|--help)
            cat << EOF
用法: $0 [选项]

选项:
    --model-type TYPE          模型类型: biomedclip, clip, pmcclip, pubmedclip (默认: biomedclip)
    --clip-backbone BACKBONE    CLIP/PubMedCLIP backbone: ViT-B/16, ViT-B/32, RN50, RN101 (默认: ViT-B/16)
    --data-dir DIR             数据目录 (默认: /home/ln/wangweicheng/ModelsTotrain/single_label_data)
    --output-dir DIR           输出目录 (默认: output)
    --gpu-id ID                GPU ID (默认: 2)
    --batch-size SIZE          批次大小 (默认: 32)
    --epochs N                 训练轮数 (默认: 100)
    --n-splits N               交叉验证折数 (默认: 5)
    --early-stopping-patience N 早停耐心值 (默认: 10)
    --class-texts-file FILE    类别文本描述 JSON 文件（仅 biomedclip 支持）
    --no-weighted-sampling     禁用加权采样
    --freeze-image-encoder     冻结图像编码器

示例:
    # 训练 BiomedCLIP
    $0 --model-type biomedclip --gpu-id 2

    # 训练标准 CLIP
    $0 --model-type clip --clip-backbone ViT-B/16 --gpu-id 2

    # 训练 PMC-CLIP
    $0 --model-type pmcclip --gpu-id 2

    # 训练 PubMedCLIP
    $0 --model-type pubmedclip --clip-backbone ViT-B/32 --gpu-id 2

    # 使用类别文本描述训练 BiomedCLIP
    $0 --model-type biomedclip --class-texts-file class_texts_hip_prosthesis.json --gpu-id 2
EOF
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证模型类型
if [[ ! "$MODEL_TYPE" =~ ^(biomedclip|clip|pmcclip|pubmedclip)$ ]]; then
    echo "错误: 无效的模型类型: $MODEL_TYPE"
    echo "支持的模型类型: biomedclip, clip, pmcclip, pubmedclip"
    exit 1
fi

# 设置 Hugging Face 环境变量（自动检测）
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_DOWNLOAD_MAX_RETRIES=10

# 检查模型是否已缓存（仅对 biomedclip）
if [[ "$MODEL_TYPE" == "biomedclip" ]]; then
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
fi

# 检查 PMC-CLIP 和 PubMedCLIP 的预训练模型文件
if [[ "$MODEL_TYPE" == "pmcclip" ]]; then
    CHECKPOINT_DIR="clip/checkpoints"
    REQUIRED_FILES=(
        "text_encoder.pth"
        "image_encoder(resnet50).pth"
        "text_projection_layer.pth"
    )
    MISSING_FILES=()
    for file in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "${CHECKPOINT_DIR}/${file}" ]; then
            MISSING_FILES+=("${file}")
        fi
    done
    if [ ${#MISSING_FILES[@]} -gt 0 ]; then
        echo "错误: PMC-CLIP 预训练模型文件缺失:"
        for file in "${MISSING_FILES[@]}"; do
            echo "  - ${CHECKPOINT_DIR}/${file}"
        done
        echo ""
        echo "请先运行以下命令下载预训练模型:"
        echo "  python download_coop_models.py"
        echo "  或"
        echo "  bash download_pretrained_models.sh"
        exit 1
    fi
fi

if [[ "$MODEL_TYPE" == "pubmedclip" ]]; then
    CHECKPOINT_DIR="clip/checkpoints"
    REQUIRED_FILE="PubMedCLIP_ViT32.pth"
    if [ ! -f "${CHECKPOINT_DIR}/${REQUIRED_FILE}" ]; then
        echo "错误: PubMedCLIP 预训练模型文件缺失:"
        echo "  - ${CHECKPOINT_DIR}/${REQUIRED_FILE}"
        echo ""
        echo "请先运行以下命令下载预训练模型:"
        echo "  python download_coop_models.py"
        echo "  或"
        echo "  bash download_pretrained_models.sh"
        exit 1
    fi
fi

# 根据模型类型设置输出目录
case "$MODEL_TYPE" in
    biomedclip)
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/biomedclip_contrastive"
        ;;
    clip)
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/clip_contrastive_${CLIP_BACKBONE//\//_}"
        ;;
    pmcclip)
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/pmcclip_contrastive"
        ;;
    pubmedclip)
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/pubmedclip_contrastive_${CLIP_BACKBONE//\//_}"
        ;;
esac

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 构建训练命令
CMD="python train_biomedcoop.py \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --model-type ${MODEL_TYPE} \
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

# 添加 CLIP backbone（仅对 clip 和 pubmedclip）
if [[ "$MODEL_TYPE" == "clip" || "$MODEL_TYPE" == "pubmedclip" ]]; then
    CMD="${CMD} --clip-backbone ${CLIP_BACKBONE}"
fi

# 添加类别文本描述文件（如果提供，仅对 biomedclip）
if [[ -n "${CLASS_TEXTS_FILE}" && "$MODEL_TYPE" == "biomedclip" ]]; then
    CMD="${CMD} --class-texts-file ${CLASS_TEXTS_FILE}"
fi

# 添加加权采样
if [[ "${USE_WEIGHTED_SAMPLING}" == true ]]; then
    CMD="${CMD} --use-weighted-sampling --weight-method ${WEIGHT_METHOD} --weight-smooth-factor ${WEIGHT_SMOOTH_FACTOR}"
fi

# 添加冻结图像编码器
if [[ "${FREEZE_IMAGE_ENCODER}" == true ]]; then
    CMD="${CMD} --freeze-image-encoder"
fi

# 打印配置信息
echo "============================================================"
echo "训练配置"
echo "============================================================"
echo "模型类型: ${MODEL_TYPE}"
if [[ "$MODEL_TYPE" == "clip" || "$MODEL_TYPE" == "pubmedclip" ]]; then
    echo "CLIP Backbone: ${CLIP_BACKBONE}"
fi
echo "数据目录: ${DATA_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "批次大小: ${BATCH_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LEARNING_RATE}"
echo "交叉验证折数: ${N_SPLITS}"
echo "加权采样: ${USE_WEIGHTED_SAMPLING}"
if [[ "${USE_WEIGHTED_SAMPLING}" == true ]]; then
    echo "  权重方法: ${WEIGHT_METHOD}"
    echo "  平滑因子: ${WEIGHT_SMOOTH_FACTOR}"
fi
echo "冻结图像编码器: ${FREEZE_IMAGE_ENCODER}"
echo "早停监控: ${EARLY_STOPPING_MONITOR}"
echo "早停耐心值: ${EARLY_STOPPING_PATIENCE}"
if [[ -n "${CLASS_TEXTS_FILE}" && "$MODEL_TYPE" == "biomedclip" ]]; then
    echo "类别文本描述文件: ${CLASS_TEXTS_FILE}"
fi
echo ""
echo "注意："
echo "  - 模型使用原始架构，仅使用对比损失（CLIP loss）"
echo "  - 只训练图像编码器，文本编码器被冻结"
echo "  - 不使用 CoOp prompt learning"
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

