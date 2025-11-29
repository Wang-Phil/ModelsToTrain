#!/bin/bash

# 多模型五折交叉验证顺序训练脚本（更稳定的版本）
# 按顺序在不同GPU上运行多个模型的交叉验证

set -e

# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出基础目录
OUTPUT_BASE_DIR="checkpoints/cv_multi_models"

# GPU列表（用空格分隔）
GPUS=(0 1 2 3)

# 模型列表
MODELS=(
    "resnet18"
    "resnet50"
    "convnextv2_tiny"
)

# 训练参数
EPOCHS=50
BATCH_SIZE=32
LR=0.001
OPTIMIZER="adam"
LOSS="focal"
FOCAL_GAMMA=2.5
AUGMENTATION="strong"
WEIGHT_DECAY=0.001
N_SPLITS=5
SEED=42
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_MIN_DELTA=0.1
USE_PRETRAINED=false

# 日志目录
LOG_DIR="logs/cv_multi_models"
mkdir -p "$LOG_DIR"

# ============================================

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 检查GPU
check_gpu() {
    nvidia-smi -i $1 &>/dev/null
}

# 运行单个模型的交叉验证
run_cv_for_model() {
    local model=$1
    local gpu_id=$2
    local output_dir="${OUTPUT_BASE_DIR}/${model}"
    local log_file="${LOG_DIR}/${model}_gpu${gpu_id}.log"
    
    print_info "训练模型: $model (GPU: $gpu_id)"
    
    local cmd="python train_cross_validation.py"
    cmd="$cmd --data-dir $DATA_DIR"
    cmd="$cmd --model $model"
    cmd="$cmd --n-splits $N_SPLITS"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
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
    
    [ "$USE_PRETRAINED" = true ] && cmd="$cmd --pretrained"
    
    print_info "执行: $cmd"
    $cmd 2>&1 | tee "$log_file"
    
    return ${PIPESTATUS[0]}
}

# 主函数
main() {
    print_info "=========================================="
    print_info "多模型五折交叉验证顺序训练"
    print_info "=========================================="
    print_info "模型: ${MODELS[@]}"
    print_info "GPU: ${GPUS[@]}"
    print_info "=========================================="
    
    # 检查GPU
    local available_gpus=()
    for gpu in "${GPUS[@]}"; do
        if check_gpu $gpu; then
            available_gpus+=($gpu)
            print_success "GPU $gpu 可用"
        else
            print_warning "GPU $gpu 不可用"
        fi
    done
    
    [ ${#available_gpus[@]} -eq 0 ] && { print_error "没有可用GPU！"; exit 1; }
    
    mkdir -p "$OUTPUT_BASE_DIR" "$LOG_DIR"
    
    local success_models=()
    local failed_models=()
    local model_idx=0
    
    for model in "${MODELS[@]}"; do
        local gpu_idx=$((model_idx % ${#available_gpus[@]}))
        local gpu=${available_gpus[$gpu_idx]}
        
        print_info "----------------------------------------"
        print_info "训练模型 $((model_idx + 1))/${#MODELS[@]}: $model"
        print_info "使用GPU: $gpu"
        print_info "----------------------------------------"
        
        if run_cv_for_model "$model" "$gpu"; then
            print_success "模型 $model 训练完成"
            success_models+=($model)
        else
            print_error "模型 $model 训练失败"
            failed_models+=($model)
        fi
        
        model_idx=$((model_idx + 1))
    done
    
    # 总结
    print_info "=========================================="
    print_info "训练总结"
    print_info "=========================================="
    print_success "成功: ${#success_models[@]} - ${success_models[@]}"
    [ ${#failed_models[@]} -gt 0 ] && print_error "失败: ${#failed_models[@]} - ${failed_models[@]}"
    
    # 生成报告
    print_info "生成汇总报告..."
    python3 << EOF
import json
import os
from pathlib import Path

output_base = "${OUTPUT_BASE_DIR}"
models = [$(printf '"%s",' "${MODELS[@]}" | sed 's/,$//')]

report_file = f"{output_base}/summary_report.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 50 + "\\n")
    f.write("多模型五折交叉验证汇总报告\\n")
    f.write("=" * 50 + "\\n\\n")
    
    for model in models:
        summary_file = f"{output_base}/{model}/cv_summary.json"
        f.write(f"\\n模型: {model}\\n")
        f.write("-" * 50 + "\\n")
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file, 'r') as sf:
                    data = json.load(sf)
                f.write(f"  平均mAP: {data.get('average_mAP', 0):.2f}% ± {data.get('std_mAP', 0):.2f}%\\n")
                f.write(f"  平均Precision: {data.get('average_precision', 0):.2f}% ± {data.get('std_precision', 0):.2f}%\\n")
                f.write(f"  平均Recall: {data.get('average_recall', 0):.2f}% ± {data.get('std_recall', 0):.2f}%\\n")
                f.write(f"  平均F1: {data.get('average_f1', 0):.2f}% ± {data.get('std_f1', 0):.2f}%\\n")
                f.write(f"  平均准确率: {data.get('average_best_val_acc', 0):.2f}% ± {data.get('std_best_val_acc', 0):.2f}%\\n")
                f.write(f"  参数量: {data.get('params_millions', 0):.2f}M\\n")
                f.write(f"  FLOPs: {data.get('flops_millions', 0):.2f}M\\n")
            except Exception as e:
                f.write(f"  错误: {e}\\n")
        else:
            f.write("  状态: 未完成\\n")

print(f"报告已保存到: {report_file}")
EOF
    
    print_success "完成！"
}

main "$@"

