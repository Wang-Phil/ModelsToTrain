#!/bin/bash

# 训练使用类别文本描述的对比学习模型
# 支持 CLIP、PMC-CLIP、PubMedCLIP 和 BiomedCLIP

set -e

# 默认参数
MODEL_TYPE=""
CLIP_BACKBONE="ViT-B/16"
DATA_DIR="/home/ln/wangweicheng/ModelsTotrain/single_label_data"
OUTPUT_DIR=""
GPU_ID=0
N_SPLITS=5
EPOCHS=100
EARLY_STOPPING_MONITOR="val_mAP"
EARLY_STOPPING_PATIENCE=100
USE_WEIGHTED_SAMPLING=true
WEIGHT_METHOD="inverse_freq"
WEIGHT_SMOOTH_FACTOR=1.0
CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"

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
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --n-splits)
            N_SPLITS="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --early-stopping-monitor)
            EARLY_STOPPING_MONITOR="$2"
            shift 2
            ;;
        --early-stopping-patience)
            EARLY_STOPPING_PATIENCE="$2"
            shift 2
            ;;
        --use-weighted-sampling)
            USE_WEIGHTED_SAMPLING=true
            shift
            ;;
        --no-weighted-sampling)
            USE_WEIGHTED_SAMPLING=false
            shift
            ;;
        --weight-method)
            WEIGHT_METHOD="$2"
            shift 2
            ;;
        --weight-smooth-factor)
            WEIGHT_SMOOTH_FACTOR="$2"
            shift 2
            ;;
        --class-texts-file)
            CLASS_TEXTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model-type TYPE          模型类型: clip, pmcclip, pubmedclip, biomedclip (必需)"
            echo "  --clip-backbone BACKBONE   CLIP/PubMedCLIP backbone (默认: ViT-B/16)"
            echo "  --data-dir DIR             数据目录 (默认: /home/ln/wangweicheng/ModelsTotrain/single_label_data)"
            echo "  --output-dir DIR           输出目录 (必需)"
            echo "  --gpu-id ID               GPU ID (默认: 0)"
            echo "  --n-splits N              交叉验证折数 (默认: 5)"
            echo "  --epochs N                训练轮数 (默认: 100)"
            echo "  --early-stopping-monitor  早停监控指标: val_acc, val_loss, val_mAP (默认: val_mAP)"
            echo "  --early-stopping-patience 早停耐心值 (默认: 100)"
            echo "  --use-weighted-sampling   启用加权采样"
            echo "  --no-weighted-sampling    禁用加权采样"
            echo "  --weight-method METHOD     权重计算方法: inverse_freq, balanced, inverse_sqrt (默认: inverse_freq)"
            echo "  --weight-smooth-factor    权重平滑因子 (默认: 1.0)"
            echo "  --class-texts-file FILE   类别文本描述 JSON 文件 (默认: class_texts_hip_prosthesis.json)"
            echo ""
            echo "示例:"
            echo "  $0 --model-type clip --output-dir output/clip_with_texts --gpu-id 2"
            echo "  $0 --model-type pmcclip --output-dir output/pmcclip_with_texts --gpu-id 2"
            echo "  $0 --model-type pubmedclip --clip-backbone ViT-B/32 --output-dir output/pubmedclip_with_texts --gpu-id 2"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$MODEL_TYPE" ]; then
    echo "错误: 必须指定 --model-type"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo "错误: 必须指定 --output-dir"
    exit 1
fi

# 根据模型类型设置默认输出目录
if [ -z "$OUTPUT_DIR" ]; then
    case $MODEL_TYPE in
        clip)
            OUTPUT_DIR="output/clip_contrastive_${CLIP_BACKBONE//\//_}_with_texts"
            ;;
        pmcclip)
            OUTPUT_DIR="output/pmcclip_contrastive_with_texts"
            ;;
        pubmedclip)
            OUTPUT_DIR="output/pubmedclip_contrastive_${CLIP_BACKBONE//\//_}_with_texts"
            ;;
        biomedclip)
            OUTPUT_DIR="output/biomedclip_contrastive_with_texts"
            ;;
        *)
            OUTPUT_DIR="output/${MODEL_TYPE}_with_texts"
            ;;
    esac
fi

# 检查 JSON 文件是否存在
if [ ! -z "$CLASS_TEXTS_FILE" ] && [ ! -f "$CLASS_TEXTS_FILE" ]; then
    # 尝试在项目根目录查找
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ -f "$SCRIPT_DIR/$CLASS_TEXTS_FILE" ]; then
        CLASS_TEXTS_FILE="$SCRIPT_DIR/$CLASS_TEXTS_FILE"
    else
        echo "警告: 类别文本描述文件不存在: $CLASS_TEXTS_FILE"
        echo "将使用默认 prompt"
    fi
fi

# 构建训练命令
CMD="python train_biomedcoop.py"
CMD="${CMD} --data-dir ${DATA_DIR}"
CMD="${CMD} --output-dir ${OUTPUT_DIR}"
CMD="${CMD} --model-type ${MODEL_TYPE}"
CMD="${CMD} --batch-size 32"
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --learning-rate 1e-4"
CMD="${CMD} --weight-decay 0.01"
CMD="${CMD} --img-size 224"
CMD="${CMD} --augmentation standard"
CMD="${CMD} --num-workers 4"
CMD="${CMD} --gpu-id ${GPU_ID}"
CMD="${CMD} --n-splits ${N_SPLITS}"
CMD="${CMD} --random-state 42"
CMD="${CMD} --early-stopping-patience ${EARLY_STOPPING_PATIENCE}"
CMD="${CMD} --early-stopping-min-delta 0.001"
CMD="${CMD} --early-stopping-monitor ${EARLY_STOPPING_MONITOR}"

# 添加加权采样参数
if [ "$USE_WEIGHTED_SAMPLING" = true ]; then
    CMD="${CMD} --use-weighted-sampling"
    CMD="${CMD} --weight-method ${WEIGHT_METHOD}"
    CMD="${CMD} --weight-smooth-factor ${WEIGHT_SMOOTH_FACTOR}"
fi

# 添加类别文本描述文件
if [ ! -z "$CLASS_TEXTS_FILE" ]; then
    CMD="${CMD} --class-texts-file ${CLASS_TEXTS_FILE}"
fi

# 添加 CLIP backbone（仅对 CLIP 和 PubMedCLIP）
if [ "$MODEL_TYPE" = "clip" ] || [ "$MODEL_TYPE" = "pubmedclip" ]; then
    CMD="${CMD} --clip-backbone ${CLIP_BACKBONE}"
fi

# 显示配置
echo "============================================================"
echo "训练配置（使用类别文本描述）"
echo "============================================================"
echo "模型类型: $MODEL_TYPE"
if [ "$MODEL_TYPE" = "clip" ] || [ "$MODEL_TYPE" = "pubmedclip" ]; then
    echo "CLIP Backbone: $CLIP_BACKBONE"
fi
echo "数据目录: $DATA_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "批次大小: 32"
echo "训练轮数: $EPOCHS"
echo "学习率: 1e-4"
echo "交叉验证折数: $N_SPLITS"
echo "加权采样: $USE_WEIGHTED_SAMPLING"
if [ "$USE_WEIGHTED_SAMPLING" = true ]; then
    echo "  权重方法: $WEIGHT_METHOD"
    echo "  平滑因子: $WEIGHT_SMOOTH_FACTOR"
fi
echo "冻结图像编码器: false"
echo "早停监控: $EARLY_STOPPING_MONITOR"
echo "早停耐心值: $EARLY_STOPPING_PATIENCE"
echo "类别文本描述文件: ${CLASS_TEXTS_FILE:-未指定}"
echo ""
echo "注意："
echo "  - 模型使用原始架构，仅使用对比损失（CLIP loss）"
echo "  - 只训练图像编码器，文本编码器被冻结"
echo "  - 不使用 CoOp prompt learning"
echo "  - 使用类别文本描述作为文本编码器输入"
echo "============================================================"
echo ""

# 执行训练
echo "执行命令:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "============================================================"
echo "✓ 训练完成！"
echo "============================================================"
echo "结果保存在: $OUTPUT_DIR"
echo "============================================================"
