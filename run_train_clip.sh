#!/bin/bash

# CLIP模型训练脚本
# 支持单个配置或多个配置组合训练
#
# StarNet 模型使用说明:
# ====================
# 1. 基本使用: 直接使用模型名称，如 "starnet_s1"
# 2. 使用预训练权重: 添加 ":pretrained" 后缀，如 "starnet_s1:pretrained"
# 3. 支持的模型:
#    - starnet_s1 (2.9M 参数, 425M FLOPs) - 最快，适合资源受限
#    - starnet_s2 (3.7M 参数, 547M FLOPs) - 平衡性能和效率
#    - starnet_s3 (5.8M 参数, 757M FLOPs) - 更高精度
#    - starnet_s4 (7.5M 参数, 1075M FLOPs) - 最高精度
#    - starnet_s050, starnet_s100, starnet_s150 - 超小模型
# 4. 预训练权重: StarNet S1-S4 有 ImageNet-1k 预训练权重可用
#
# 示例配置:
# ========
# 单模型训练: 设置 USE_MULTI_CONFIG=false，并在 IMAGE_ENCODERS 中只保留一个模型
# 多模型对比: 设置 USE_MULTI_CONFIG=true，在 IMAGE_ENCODERS 中添加多个模型
# 使用预训练: 将 "starnet_s1" 改为 "starnet_s1:pretrained"

set -e

# 设置环境变量以避免警告
export TOKENIZERS_PARALLELISM=false

# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出基础目录
OUTPUT_BASE_DIR="checkpoints/clip_models"

# 日志目录
LOG_DIR="logs/clip_training"
mkdir -p "$LOG_DIR"

# GPU列表（用空格分隔，例如: "0 1 2 3"）
# 如果只有一个GPU，可以设置为单个值，如: GPUS=(5)
# 多个GPU会循环分配给不同配置
GPUS=(5)

# 训练参数
BATCH_SIZE=32
EPOCHS=200
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
EMBED_DIM=512
TEMPERATURE=0.07
IMG_SIZE=224
AUGMENTATION="standard"
NUM_WORKERS=4
USE_AMP=true

# 图像编码器列表
# 支持的 StarNet 模型: starnet_s1, starnet_s2, starnet_s3, starnet_s4, starnet_s050, starnet_s100, starnet_s150
# 使用预训练权重: starnet_s1:pretrained 或 starnet_s2:true
IMAGE_ENCODERS=(
    # StarNet 模型系列（推荐）
    "starnet_s1"              # 2.9M 参数，最快
    # "starnet_s1:pretrained" # 使用 ImageNet 预训练权重
    # "starnet_s2"            # 3.7M 参数，平衡
    # "starnet_s2:pretrained" # 使用 ImageNet 预训练权重
    # "starnet_s3"            # 5.8M 参数，更高精度
    # "starnet_s4"            # 7.5M 参数，最高精度
    
    # 其他模型
    # "starnet_dual_pyramid_rcf"  # StarNet Dual-Pyramid RCF 变体
    # "resnet18"
    # "resnet50"
    # "efficientnet-b0"
)

# 文本编码器列表
TEXT_ENCODERS=(
    "clip:ViT-B/32"
)

# 是否使用多配置训练（true/false）
USE_MULTI_CONFIG=true

# 配置文件路径（如果USE_MULTI_CONFIG=true且CONFIG_FILE不为空，则使用配置文件）
CONFIG_FILE="train_clip_config.json"

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

# 检查GPU是否可用
check_gpu() {
    local gpu_id=$1
    if ! nvidia-smi -i $gpu_id &>/dev/null; then
        return 1
    fi
    return 0
}

# 等待GPU可用
wait_for_gpu() {
    local gpu_id=$1
    local max_wait=300  # 最多等待5分钟
    local waited=0
    
    while [ $waited -lt $max_wait ]; do
        # 检查GPU是否可用（使用率低于70%）
        local utilization=$(nvidia-smi -i $gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        if [ -z "$utilization" ]; then
            # 如果无法获取使用率，直接返回（可能GPU空闲）
            return 0
        fi
        
        if [ "$utilization" -lt 70 ]; then
            return 0
        fi
        
        print_info "GPU $gpu_id 使用中（使用率: ${utilization}%），等待中..."
        sleep 10
        waited=$((waited + 10))
    done
    
    print_warning "GPU $gpu_id 等待超时，继续执行..."
    return 0
}

# 运行单个配置的训练
run_single_config() {
    local img_encoder=$1
    local txt_encoder=$2
    local gpu_id=$3
    
    # 清理编码器名称中的特殊字符（如冒号）
    local img_enc_clean=$(echo "$img_encoder" | tr ':/' '_')
    local txt_enc_clean=$(echo "$txt_encoder" | tr ':/' '_')
    local config_name="${img_enc_clean}_${txt_enc_clean}"
    
    local output_dir="$OUTPUT_BASE_DIR/${config_name}"
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/${config_name}_gpu${gpu_id}_${timestamp}.log"
    
    print_info "开始训练配置: $config_name (GPU: $gpu_id)"
    print_info "输出目录: $output_dir"
    print_info "日志文件: $log_file"
    
    # 将开始信息写入日志文件
    {
        echo "=========================================="
        echo "CLIP模型单配置训练"
        echo "开始时间: $(date)"
        echo "=========================================="
        echo "数据目录: $DATA_DIR"
        echo "输出目录: $output_dir"
        echo "图像编码器: $img_encoder"
        echo "文本编码器: $txt_encoder"
        echo "GPU ID: $gpu_id"
        echo ""
    } > "$log_file"
    
    # 构建训练命令
    local cmd="python train_clip.py"
    cmd="$cmd --data-dir $DATA_DIR"
    cmd="$cmd --output-dir $output_dir"
    cmd="$cmd --image-encoder $img_encoder"
    cmd="$cmd --text-encoder $txt_encoder"
    cmd="$cmd --embed-dim $EMBED_DIM"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --learning-rate $LEARNING_RATE"
    cmd="$cmd --weight-decay $WEIGHT_DECAY"
    cmd="$cmd --temperature $TEMPERATURE"
    cmd="$cmd --img-size $IMG_SIZE"
    cmd="$cmd --augmentation $AUGMENTATION"
    cmd="$cmd --num-workers $NUM_WORKERS"
    cmd="$cmd --gpu-id $gpu_id"
    
    if [ "$USE_AMP" = true ]; then
        cmd="$cmd --use-amp"
    fi
    
    # 运行训练并记录日志
    print_info "执行命令: $cmd"
    $cmd >> "$log_file" 2>&1
    
    local exit_code=$?
    
    # 写入结束信息
    {
        echo ""
        echo "=========================================="
        echo "结束时间: $(date)"
        if [ $exit_code -eq 0 ]; then
            echo "状态: 成功完成"
        else
            echo "状态: 失败 (退出码: $exit_code)"
        fi
        echo "=========================================="
    } >> "$log_file"
    
    if [ $exit_code -eq 0 ]; then
        print_success "配置 $config_name 训练完成 (GPU: $gpu_id)"
        return 0
    else
        print_error "配置 $config_name 训练失败 (GPU: $gpu_id, 退出码: $exit_code)"
        print_error "查看日志: $log_file"
        return 1
    fi
}

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    print_error "数据目录不存在: $DATA_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_BASE_DIR"

print_info "数据目录: $DATA_DIR"
print_info "输出目录: $OUTPUT_BASE_DIR"
print_info "日志目录: $LOG_DIR"
print_info "GPU列表: ${GPUS[@]}"

if [ "$USE_MULTI_CONFIG" = true ]; then
    print_info "使用多配置并行训练模式"
    
    # 检查GPU可用性
    print_info "检查GPU可用性..."
    local available_gpus=()
    for gpu in "${GPUS[@]}"; do
        if check_gpu $gpu; then
            available_gpus+=($gpu)
            print_success "GPU $gpu 可用"
        else
            print_warning "GPU $gpu 不可用，跳过"
        fi
    done
    
    if [ ${#available_gpus[@]} -eq 0 ]; then
        print_error "没有可用的GPU！"
        exit 1
    fi
    
    print_info "可用GPU: ${available_gpus[@]}"
    
    # 生成所有配置组合
    local configs=()
    if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
        print_info "从配置文件读取配置: $CONFIG_FILE"
        # 从JSON文件读取配置
        while IFS='|' read -r img_enc txt_enc; do
            if [ -n "$img_enc" ] && [ -n "$txt_enc" ]; then
                configs+=("${img_enc}|${txt_enc}")
            fi
        done < <(python3 << 'PYTHON_EOF'
import json
import sys

try:
    with open(sys.argv[1], 'r') as f:
        configs = json.load(f)
    for config in configs:
        img_enc = config.get('image_encoder_name', config.get('image_encoder', 'unknown'))
        txt_enc = config.get('text_encoder_name', config.get('text_encoder', 'unknown'))
        print(f"{img_enc}|{txt_enc}")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_EOF
"$CONFIG_FILE")
    else
        print_info "使用默认配置组合"
        # 生成所有配置组合
        for img_enc in "${IMAGE_ENCODERS[@]}"; do
            for txt_enc in "${TEXT_ENCODERS[@]}"; do
                configs+=("${img_enc}|${txt_enc}")
            done
        done
    fi
    
    print_info "总共 ${#configs[@]} 个配置需要训练"
    
    # 创建输出和日志目录
    mkdir -p "$OUTPUT_BASE_DIR"
    mkdir -p "$LOG_DIR"
    
    # 存储所有后台进程的PID
    declare -A pids
    declare -A gpu_assignments
    declare -A config_names
    
    # 为每个配置分配GPU并启动训练
    local config_idx=0
    for config_pair in "${configs[@]}"; do
        IFS='|' read -r img_encoder txt_encoder <<< "$config_pair"
        
        # 循环分配GPU
        local gpu_idx=$((config_idx % ${#available_gpus[@]}))
        local assigned_gpu=${available_gpus[$gpu_idx]}
        
        # 清理编码器名称
        local img_enc_clean=$(echo "$img_encoder" | tr ':/' '_')
        local txt_enc_clean=$(echo "$txt_encoder" | tr ':/' '_')
        local config_name="${img_enc_clean}_${txt_enc_clean}"
        
        print_info "为配置 $config_name 分配 GPU $assigned_gpu"
        
        # 等待GPU可用
        wait_for_gpu $assigned_gpu
        
        # 在后台运行训练
        run_single_config "$img_encoder" "$txt_encoder" "$assigned_gpu" &
        local pid=$!
        pids[$config_name]=$pid
        gpu_assignments[$config_name]=$assigned_gpu
        config_names[$config_name]="$img_encoder|$txt_encoder"
        
        print_info "配置 $config_name 已在后台启动 (PID: $pid, GPU: $assigned_gpu)"
        
        # 稍微延迟，避免同时启动太多进程
        sleep 5
        
        config_idx=$((config_idx + 1))
    done
    
    # 等待所有进程完成
    print_info "=========================================="
    print_info "等待所有训练任务完成..."
    print_info "=========================================="
    
    local failed_configs=()
    local success_configs=()
    
    for config_name in "${!pids[@]}"; do
        local pid=${pids[$config_name]}
        local gpu=${gpu_assignments[$config_name]}
        
        print_info "等待配置 $config_name 完成 (PID: $pid, GPU: $gpu)..."
        
        if wait $pid; then
            print_success "配置 $config_name 训练成功完成"
            success_configs+=($config_name)
        else
            print_error "配置 $config_name 训练失败"
            failed_configs+=($config_name)
        fi
    done
    
    # 打印总结
    print_info "=========================================="
    print_info "训练总结"
    print_info "=========================================="
    print_success "成功完成的配置 (${#success_configs[@]}): ${success_configs[@]}"
    
    if [ ${#failed_configs[@]} -gt 0 ]; then
        print_error "失败的配置 (${#failed_configs[@]}): ${failed_configs[@]}"
        print_error "请查看日志文件: $LOG_DIR"
        exit 1
    fi
    
    print_info "=========================================="
    print_success "所有任务完成！"
    print_info "=========================================="
else
    print_info "使用单配置训练模式"
    
    IMAGE_ENCODER="${IMAGE_ENCODERS[0]}"
    TEXT_ENCODER="${TEXT_ENCODERS[0]}"
    ASSIGNED_GPU="${GPUS[0]}"
    
    # 检查GPU可用性
    if ! check_gpu $ASSIGNED_GPU; then
        print_error "GPU $ASSIGNED_GPU 不可用！"
        exit 1
    fi
    
    # 等待GPU可用
    wait_for_gpu $ASSIGNED_GPU
    
    # 使用统一的训练函数
    run_single_config "$IMAGE_ENCODER" "$TEXT_ENCODER" "$ASSIGNED_GPU"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        exit $EXIT_CODE
    fi
fi

