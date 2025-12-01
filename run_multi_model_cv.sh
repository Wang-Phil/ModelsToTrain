#!/bin/bash

# 多模型五折交叉验证并行训练脚本
# 支持在不同GPU上并行运行多个模型的交叉验证

set -e  # 遇到错误立即退出

# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出基础目录
OUTPUT_BASE_DIR="checkpoints/final_starnet_models/final_model"

# GPU列表（用空格分隔，例如: "0 1 2 3"）
GPUS=(8 1 5 6 7)

# PLD setting for long tail 
#===============================
# LOSS="ldam"
# LDAM_MARGIN=0.5
# LDAM_S=30
# USE_DRW=true
# DRW_START_EPOCH=60
# DRW_BETA=0.9999
# DRW_WINDOW=10
# DRW_THRESHOLD=0.05

# 模型列表（要训练的模型名称）
MODELS=(
    # 空间注意力机制换一个位置尝试实验
    # "starnet_s2"
    # 统一归一化，同时优化权重配置
    # "starnet_dual_pyramid_sa"
    # 去除了空间注意力机制，再试一下
    # "starnet_dual_pyramid"
    # 双金字塔，使用Swin Transformer+空间注意力机制
    # "starnet_dual_swin_pyramid"
    # 原始StarNet
    # "starnet_s1"
    # 基于FPN的StarNet
    # "starnet_hybrid_s"
    # "starnet_hybrid_t"
    # "starnet_vit_hybrid_s"
    # "starnet_vit_hybrid_t"
    # "mobilenetv3_small"
    # "starnet_s1"  
    # "starnet_s2"
    # "starnet_s3"
    # "resnet18"
    # "resnet50"
    # "resnet101"
    # "inceptionv3"
    # "densenet121"
    # "densenet161"
    # "mobilenetv2"
    # "googlenet"
    # "starnet_dual_swin_pyramid"
    # "lsnet_t"
    # "lsnet_s"
    # "lsnet_b"
    # "starnet_vit_hybrid_t"
    # 重新测试
    # "starnet_dual_swin_pyramid"
    # "starnet_s1_pyramid"
    # "starnet_s1_parallel_sa"
    # "starnet_arconv_s1"
    # "starnet_dual_pyramid_rcf"
    # "starnet_s1_gated"
    # "starnet_s1_odconv"
    # "starnet_s1_lsk"
    # "starnet_s1_grn"
    # "starnet_gated_s1"
    # "starnet_s1_gated_skip"
    # "starnet_s1_dl"
    # "starnet_s1_lora"

    # 验证cross attention与空间注意力融合效果
    # "starnet_s1_final"
    # "starnet_s2_final"
    # "starnet_s3_final"

    # 空间注意力消融实验 (Spatial Attention Variants)
    "starnet_sa_s1"  # 所有stage都加空间注意力 (stage 0,1,2,3)
    # "starnet_sa_s2"  # 第一个stage不加注意力 (stage 1,2,3加注意力)
    # "starnet_sa_s3"  # 前两个stage不加注意力 (stage 2,3加注意力)
    # "starnet_sa_s4"  # 前三个stage不加注意力 (只有stage 3加注意力)

    # 交叉星乘 消融实验
    # "starnet_s1_cross_star"
    # "starnet_s1_cross_star_add"
    # "starnet_s1_cross_star_samescale"

    # 调整gln与注意力的位置
    # "starnet_s1_cross_with_gln"

    # 新模型
    # "starnet_cf_s3"

    # 空间注意力机制消融实验
    # "starnet_s1"
    # "starnet_s2"
    # "starnet_s3"
    # "starnet_s4"

)

# 训练参数
EPOCHS=200
BATCH_SIZE=32
LR=0.001
OPTIMIZER="adam"
LOSS="focal"
FOCAL_GAMMA=2
AUGMENTATION="standard"
WEIGHT_DECAY=0.001
N_SPLITS=5
SEED=42
# 早停策略：基于val_mAP监控，patience默认30
EARLY_STOPPING_PATIENCE=60
EARLY_STOPPING_MIN_DELTA=0.1  # mAP提升超过0.1%才算改进

# 显存优化参数
# 梯度累积步数（用于减少显存占用，有效批次大小 = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS）
# 对于大模型（如 starnet_dual_pyramid_rcf），建议使用梯度累积
GRADIENT_ACCUMULATION_STEPS=1  # 默认不使用梯度累积
# 针对特定模型的批次大小和梯度累积配置（可选）
declare -A MODEL_BATCH_SIZES
declare -A MODEL_GRAD_ACCUM
# 示例：为显存占用大的模型配置更小的批次大小和梯度累积
MODEL_BATCH_SIZES["starnet_dual_pyramid_rcf"]=16
MODEL_GRAD_ACCUM["starnet_dual_pyramid_rcf"]=2  # 有效批次大小 = 16 × 2 = 32

# starnet_gated_s1 显存优化配置（解决内存碎片化问题）
MODEL_BATCH_SIZES["starnet_gated_s1"]=16
MODEL_GRAD_ACCUM["starnet_gated_s1"]=2  # 有效批次大小 = 16 × 2 = 32

# 快速验证模式（简单train/val划分，不使用交叉验证）
# 设置为 true 启用快速验证模式，false 使用交叉验证
USE_SIMPLE_SPLIT=false
VAL_RATIO=0.4  # 验证集比例（仅在USE_SIMPLE_SPLIT=true时使用）

# 是否使用预训练权重（true/false）
USE_PRETRAINED=false

# 日志目录
LOG_DIR="logs/final_starnet_models/final_model"
mkdir -p "$LOG_DIR"

# ============================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
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
        local utilization=$(nvidia-smi -i $gpu_id --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [ -z "$utilization" ]; then
            print_error "无法获取GPU $gpu_id 的使用率"
            return 1
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

# 运行单个模型的交叉验证
run_cv_for_model() {
    local model=$1
    local gpu_id=$2
    local output_dir="${OUTPUT_BASE_DIR}/${model}"
    local log_file="${LOG_DIR}/${model}_gpu${gpu_id}.log"
    
    print_info "开始训练模型: $model (GPU: $gpu_id)"
    print_info "输出目录: $output_dir"
    print_info "日志文件: $log_file"
    
    # 构建训练命令
    local cmd="python train_cross_validation.py"
    cmd="$cmd --data-dir $DATA_DIR"
    cmd="$cmd --model $model"
    
    # 根据模式选择参数
    if [ "$USE_SIMPLE_SPLIT" = true ]; then
        cmd="$cmd --simple-split"
        cmd="$cmd --val-ratio $VAL_RATIO"
        print_info "使用快速验证模式（简单train/val划分）"
    else
        cmd="$cmd --n-splits $N_SPLITS"
        print_info "使用交叉验证模式（$N_SPLITS折）"
    fi
    
    cmd="$cmd --epochs $EPOCHS"
    
    # 显存优化：根据模型配置批次大小和梯度累积
    local model_batch_size=$BATCH_SIZE
    local model_grad_accum=$GRADIENT_ACCUMULATION_STEPS
    
    if [ -n "${MODEL_BATCH_SIZES[$model]}" ]; then
        model_batch_size=${MODEL_BATCH_SIZES[$model]}
        print_info "模型 $model 使用自定义批次大小: $model_batch_size"
    fi
    
    if [ -n "${MODEL_GRAD_ACCUM[$model]}" ]; then
        model_grad_accum=${MODEL_GRAD_ACCUM[$model]}
        print_info "模型 $model 使用梯度累积步数: $model_grad_accum (有效批次大小: $((model_batch_size * model_grad_accum)))"
    fi
    
    cmd="$cmd --batch-size $model_batch_size"
    if [ "$model_grad_accum" -gt 1 ]; then
        cmd="$cmd --gradient-accumulation-steps $model_grad_accum"
    fi
    
    cmd="$cmd --lr $LR"
    cmd="$cmd --optimizer $OPTIMIZER"
    cmd="$cmd --loss $LOSS"
    cmd="$cmd --focal-gamma $FOCAL_GAMMA"
    cmd="$cmd --augmentation $AUGMENTATION"
    cmd="$cmd --weight-decay $WEIGHT_DECAY"
    cmd="$cmd --early-stopping-patience $EARLY_STOPPING_PATIENCE"
    cmd="$cmd --early-stopping-min-delta $EARLY_STOPPING_MIN_DELTA"
    cmd="$cmd --output-dir $output_dir"
    cmd="$cmd --device cuda:$gpu_id"
    cmd="$cmd --seed $SEED"
    
    # 显存优化：根据模型配置批次大小和梯度累积
    local model_batch_size=$BATCH_SIZE
    local model_grad_accum=$GRADIENT_ACCUMULATION_STEPS
    
    if [ -n "${MODEL_BATCH_SIZES[$model]}" ]; then
        model_batch_size=${MODEL_BATCH_SIZES[$model]}
        print_info "模型 $model 使用自定义批次大小: $model_batch_size"
    fi
    
    if [ -n "${MODEL_GRAD_ACCUM[$model]}" ]; then
        model_grad_accum=${MODEL_GRAD_ACCUM[$model]}
        print_info "模型 $model 使用梯度累积步数: $model_grad_accum (有效批次大小: $((model_batch_size * model_grad_accum)))"
    fi
    
    # 更新批次大小和梯度累积参数
    cmd="$cmd --batch-size $model_batch_size"
    if [ "$model_grad_accum" -gt 1 ]; then
        cmd="$cmd --gradient-accumulation-steps $model_grad_accum"
    fi
    
    [ -n "$LDAM_MARGIN" ] && cmd="$cmd --ldam-margin $LDAM_MARGIN"
    [ -n "$LDAM_S" ] && cmd="$cmd --ldam-s $LDAM_S"
    if [ "$USE_DRW" = true ]; then
        cmd="$cmd --use-drw --drw-start-epoch $DRW_START_EPOCH --drw-beta $DRW_BETA"
    fi

    if [ "$USE_PRETRAINED" = true ]; then
        cmd="$cmd --pretrained"
    fi
    
    # 运行训练并记录日志
    print_info "执行命令: $cmd"
    $cmd > "$log_file" 2>&1
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "模型 $model 训练完成 (GPU: $gpu_id)"
        return 0
    else
        print_error "模型 $model 训练失败 (GPU: $gpu_id, 退出码: $exit_code)"
        print_error "查看日志: $log_file"
        return 1
    fi
}

# 主函数
main() {
    print_info "=========================================="
    print_info "多模型五折交叉验证并行训练"
    print_info "=========================================="
    print_info "数据目录: $DATA_DIR"
    print_info "输出目录: $OUTPUT_BASE_DIR"
    print_info "模型列表: ${MODELS[@]}"
    print_info "GPU列表: ${GPUS[@]}"
    print_info "训练轮数: $EPOCHS"
    print_info "批次大小: $BATCH_SIZE"
    print_info "学习率: $LR"
    print_info "优化器: $OPTIMIZER"
    print_info "损失函数: $LOSS"
    print_info "数据增强: $AUGMENTATION"
    if [ "$USE_SIMPLE_SPLIT" = true ]; then
        print_info "训练模式: 快速验证（简单train/val划分，验证集比例: $VAL_RATIO）"
    else
        print_info "训练模式: 交叉验证（$N_SPLITS折）"
    fi
    print_info "=========================================="
    
    # 检查GPU
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
    
    # 创建输出和日志目录
    mkdir -p "$OUTPUT_BASE_DIR"
    mkdir -p "$LOG_DIR"
    
    # 存储所有后台进程的PID
    declare -A pids
    declare -A gpu_assignments
    
    # 为每个模型分配GPU并启动训练
    local model_idx=0
    for model in "${MODELS[@]}"; do
        # 循环分配GPU
        local gpu_idx=$((model_idx % ${#available_gpus[@]}))
        local assigned_gpu=${available_gpus[$gpu_idx]}
        
        print_info "为模型 $model 分配 GPU $assigned_gpu"
        
        # 等待GPU可用
        wait_for_gpu $assigned_gpu
        
        # 在后台运行训练
        run_cv_for_model "$model" "$assigned_gpu" &
        local pid=$!
        pids[$model]=$pid
        gpu_assignments[$model]=$assigned_gpu
        
        print_info "模型 $model 已在后台启动 (PID: $pid, GPU: $assigned_gpu)"
        
        # 稍微延迟，避免同时启动太多进程
        sleep 5
        
        model_idx=$((model_idx + 1))
    done
    
    # 等待所有进程完成
    print_info "=========================================="
    print_info "等待所有训练任务完成..."
    print_info "=========================================="
    
    local failed_models=()
    local success_models=()
    
    for model in "${MODELS[@]}"; do
        local pid=${pids[$model]}
        local gpu=${gpu_assignments[$model]}
        
        print_info "等待模型 $model 完成 (PID: $pid, GPU: $gpu)..."
        
        if wait $pid; then
            print_success "模型 $model 训练成功完成"
            success_models+=($model)
        else
            print_error "模型 $model 训练失败"
            failed_models+=($model)
        fi
    done
    
    # 打印总结
    print_info "=========================================="
    print_info "训练总结"
    print_info "=========================================="
    print_success "成功完成的模型 (${#success_models[@]}): ${success_models[@]}"
    
    if [ ${#failed_models[@]} -gt 0 ]; then
        print_error "失败的模型 (${#failed_models[@]}): ${failed_models[@]}"
        print_error "请查看日志文件: $LOG_DIR"
    fi
    
    # 生成汇总报告
    print_info "生成汇总报告..."
    generate_summary_report
    
    print_info "=========================================="
    print_success "所有任务完成！"
    print_info "=========================================="
}

# 生成汇总报告
generate_summary_report() {
    local report_file="${OUTPUT_BASE_DIR}/summary_report.txt"
    
    {
        echo "=========================================="
        if [ "$USE_SIMPLE_SPLIT" = true ]; then
            echo "多模型快速验证汇总报告"
        else
            echo "多模型五折交叉验证汇总报告"
        fi
        echo "生成时间: $(date)"
        echo "=========================================="
        echo ""
        echo "训练配置:"
        echo "  数据目录: $DATA_DIR"
        echo "  训练轮数: $EPOCHS"
        echo "  批次大小: $BATCH_SIZE"
        echo "  学习率: $LR"
        echo "  优化器: $OPTIMIZER"
        echo "  损失函数: $LOSS"
        echo "  数据增强: $AUGMENTATION"
        if [ "$USE_SIMPLE_SPLIT" = true ]; then
            echo "  训练模式: 快速验证（简单train/val划分，验证集比例: $VAL_RATIO）"
        else
            echo "  训练模式: 交叉验证（$N_SPLITS折）"
        fi
        echo ""
        echo "=========================================="
        echo "各模型结果:"
        echo "=========================================="
    } > "$report_file"
    
    for model in "${MODELS[@]}"; do
        # 根据模式选择不同的结果文件
        if [ "$USE_SIMPLE_SPLIT" = true ]; then
            local summary_file="${OUTPUT_BASE_DIR}/${model}/simple_split_summary.json"
        else
            local summary_file="${OUTPUT_BASE_DIR}/${model}/cv_summary.json"
        fi
        
        if [ -f "$summary_file" ]; then
            {
                echo ""
                echo "模型: $model"
                echo "----------------------------------------"
                python3 << EOF
import json
import sys

try:
    with open('$summary_file', 'r') as f:
        data = json.load(f)
    
    mode = data.get('mode', 'cv')
    if mode == 'simple_split':
        # 简单划分模式的结果
        if 'best_val_mAP' in data:
            print(f"  最佳验证mAP: {data.get('best_val_mAP', 0):.2f}% (Epoch {data.get('best_epoch', 0)})")
        print(f"  mAP: {data.get('mAP', 0):.2f}%")
        print(f"  Precision: {data.get('precision_macro', 0):.2f}%")
        print(f"  Recall: {data.get('recall_macro', 0):.2f}%")
        print(f"  F1 Score: {data.get('f1_macro', 0):.2f}%")
        print(f"  准确率: {data.get('best_val_acc', 0):.2f}% (Epoch {data.get('best_epoch', 0)})")
        print(f"  参数量: {data.get('params_millions', 0):.2f}M")
        print(f"  FLOPs: {data.get('flops_millions', 0):.2f}M")
    else:
        # 交叉验证模式的结果
        if 'average_best_val_mAP' in data:
            print(f"  平均最佳验证mAP: {data.get('average_best_val_mAP', 0):.2f}% ± {data.get('std_best_val_mAP', 0):.2f}%")
        print(f"  平均mAP: {data.get('average_mAP', 0):.2f}% ± {data.get('std_mAP', 0):.2f}%")
        print(f"  平均Precision: {data.get('average_precision', 0):.2f}% ± {data.get('std_precision', 0):.2f}%")
        print(f"  平均Recall: {data.get('average_recall', 0):.2f}% ± {data.get('std_recall', 0):.2f}%")
        print(f"  平均F1 Score: {data.get('average_f1', 0):.2f}% ± {data.get('std_f1', 0):.2f}%")
        print(f"  平均准确率: {data.get('average_best_val_acc', 0):.2f}% ± {data.get('std_best_val_acc', 0):.2f}%")
        print(f"  参数量: {data.get('params_millions', 0):.2f}M")
        print(f"  FLOPs: {data.get('flops_millions', 0):.2f}M")
except Exception as e:
    print(f"  错误: 无法读取结果文件 - {e}")
EOF
            } >> "$report_file"
        else
            echo "" >> "$report_file"
            echo "模型: $model" >> "$report_file"
            echo "  状态: 训练失败或未完成" >> "$report_file"
        fi
    done
    
    print_success "汇总报告已保存到: $report_file"
}

# 运行主函数
main "$@"

