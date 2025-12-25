#!/bin/bash
# 评估五折交叉验证平均结果示例

# 评估 starnet_s1 模型的五折交叉验证平均结果
echo "评估 starnet_s1 五折交叉验证平均结果..."
python evaluate_cv_average.py \
    --model-base-dir checkpoints/final_models/ablation/starnet_s1 \
    --data-dir data/test \
    --output-dir evaluation_results/starnet_s1_cv_average \
    --batch-size 32 \
    --num-workers 4

echo ""
echo "评估完成！结果保存在: evaluation_results/starnet_s1_cv_average"
echo ""
echo "输出文件："
echo "  - cv_average_metrics.json: JSON格式的平均指标"
echo "  - cv_average_metrics.csv: CSV格式的平均结果表格"
echo "  - cv_average_metrics.xlsx: Excel格式的平均结果表格（如果安装了openpyxl）"
echo "  - cv_average_comparison.png: 可视化对比图（带误差条）"

