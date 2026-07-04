#!/bin/bash

# Grad-CAM 生成进度监控脚本

echo "=========================================="
echo "Grad-CAM 生成进度监控"
echo "=========================================="
echo ""

# 检查进程状态
echo "【进程状态】"
if ps aux | grep -q "[g]enerate_cv_gradcam"; then
    echo "✓ 进程正在运行"
    ps aux | grep "[g]enerate_cv_gradcam" | awk '{print "  PID:", $2, "| CPU:", $3"% | 内存:", $4"%"}'
else
    echo "✗ 进程未运行"
fi
echo ""

# 查看日志最后几行
echo "【最新日志】"
tail -20 gradcam_all_samples.log 2>/dev/null | tail -10
echo ""

# 统计已生成的文件数
echo "【生成进度】"
OUTPUT_DIR="checkpoints/final_models/sk_size_ablation/starnet_s1_sk39/gradcam_output"
if [ -d "$OUTPUT_DIR" ]; then
    TOTAL=$(find "$OUTPUT_DIR" -name "*.png" 2>/dev/null | wc -l)
    echo "已生成热力图: $TOTAL 张"
    echo ""
    echo "各 Fold 生成情况:"
    for fold in {1..5}; do
        COUNT=$(find "$OUTPUT_DIR/fold_$fold" -name "*.png" 2>/dev/null | wc -l)
        if [ -d "$OUTPUT_DIR/fold_$fold" ]; then
            echo "  Fold $fold: $COUNT 张"
        else
            echo "  Fold $fold: 未开始或不存在"
        fi
    done
else
    echo "输出目录不存在"
fi
echo ""

# 查看各个类别的进度
if [ -d "$OUTPUT_DIR/fold_1" ]; then
    echo "【Fold 1 各类别进度】"
    for class_dir in "$OUTPUT_DIR/fold_1"/class_*; do
        if [ -d "$class_dir" ]; then
            CLASS_NAME=$(basename "$class_dir" | sed 's/class_[0-9]*_//')
            COUNT=$(find "$class_dir" -name "*.png" 2>/dev/null | wc -l)
            echo "  $CLASS_NAME: $COUNT 张"
        fi
    done
fi
echo ""

echo "=========================================="
echo "提示: 使用 'tail -f gradcam_all_samples.log' 实时查看日志"
echo "=========================================="

