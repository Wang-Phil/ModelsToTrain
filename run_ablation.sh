#!/bin/bash

# 消融实验脚本
# 对比不同组件组合的效果

set -e

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

# 默认参数
DATA_DIR="single_label_data"
OUTPUT_DIR="checkpoints/ablation_study"
LOG_DIR="logs/ablation_study"
GPU_ID=""
GPUS=""
BASE_CONFIG="train_clip_config.json"
SKIP_EXISTING=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --base-config)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --data-dir DIR          数据目录（默认: single_label_data）"
            echo "  --output-dir DIR        输出目录（默认: checkpoints/ablation_study）"
            echo "  --log-dir DIR           日志目录（默认: logs/ablation_study）"
            echo "  --gpu-id ID             单个GPU ID（默认: 0）"
            echo "  --gpus LIST             GPU ID列表，用逗号分隔（例如: 7,8,9）。如果指定，会自动并行运行"
            echo "  --base-config FILE      基础配置文件（默认: train_clip_config.json）"
            echo "  --skip-existing         跳过已有结果的实验"
            echo "  --help                  显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  # 使用单个GPU顺序运行"
            echo "  $0 --data-dir single_label_data --gpu-id 9"
            echo ""
            echo "  # 使用多个GPU并行运行（推荐）"
            echo "  $0 --data-dir single_label_data --gpus 7,8,9"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查数据目录
if [ ! -d "$DATA_DIR" ]; then
    print_error "数据目录不存在: $DATA_DIR"
    exit 1
fi

# 创建输出和日志目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

print_info "=========================================="
print_info "消融实验"
print_info "=========================================="
print_info "数据目录: $DATA_DIR"
print_info "输出目录: $OUTPUT_DIR"
print_info "日志目录: $LOG_DIR"
if [ -n "$GPUS" ]; then
    print_info "GPU 列表: $GPUS (并行运行)"
elif [ -n "$GPU_ID" ]; then
    print_info "GPU ID: $GPU_ID"
else
    print_info "GPU ID: 0 (默认)"
fi
print_info "基础配置: $BASE_CONFIG"
if [ "$SKIP_EXISTING" = true ]; then
    print_info "跳过已有结果: 是"
fi
print_info "=========================================="
echo ""

# 构建 Python 命令
CMD="python run_ablation_study.py"
CMD="$CMD --data-dir $DATA_DIR"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --log-dir $LOG_DIR"

if [ -n "$GPUS" ]; then
    CMD="$CMD --gpus $GPUS"
elif [ -n "$GPU_ID" ]; then
    CMD="$CMD --gpu-id $GPU_ID"
fi

CMD="$CMD --base-config $BASE_CONFIG"

if [ "$SKIP_EXISTING" = true ]; then
    CMD="$CMD --skip-existing"
fi

# 运行实验
print_info "开始运行消融实验..."
print_info "执行命令: $CMD"
echo ""

$CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    print_success "所有实验完成！"
    print_info "结果目录: $OUTPUT_DIR"
    print_info "消融实验表格: $OUTPUT_DIR/ablation_table.md"
    print_info "LaTeX 表格: $OUTPUT_DIR/ablation_table.tex"
    print_info "详细结果: $OUTPUT_DIR/ablation_results.json"
else
    print_error "实验失败（退出码: $EXIT_CODE）"
    exit $EXIT_CODE
fi


