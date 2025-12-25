#!/bin/bash
# 评估 starnet_s1 模型在不同类别上的性能指标示例

# 设置路径
MODEL_DIR="checkpoints/final_models/ablation/starnet_s1"
DATA_DIR="data/test"
OUTPUT_BASE="evaluation_results/starnet_s1"

# 评估单个fold（例如fold_1）
echo "评估 starnet_s1 fold_1..."
python evaluate_per_class.py \
    --model-path ${MODEL_DIR}/fold_1/best_model.pth \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_BASE}_fold1_per_class \
    --batch-size 32 \
    --num-workers 4

echo "评估完成！结果保存在: ${OUTPUT_BASE}_fold1_per_class"
echo ""
echo "输出文件："
echo "  - per_class_metrics.json: JSON格式的详细指标"
echo "  - per_class_metrics.csv: CSV格式的表格"
echo "  - per_class_metrics.xlsx: Excel格式的表格"
echo "  - per_class_comparison.png: 可视化对比图"

