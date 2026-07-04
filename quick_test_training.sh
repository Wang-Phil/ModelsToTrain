#!/bin/bash
# 快速测试训练脚本（只训练 1 个 epoch，用于验证代码是否正常）

DATA_DIR="/home/ln/wangweicheng/ModelsTotrain/single_label_data"
GPU_ID=${1:-0}

echo "============================================================"
echo "快速测试训练（1 个 epoch）"
echo "============================================================"

# 测试 CLIP 模型
echo ""
echo "[测试] CLIP 模型..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train_biomedcoop.py \
    --data-dir "$DATA_DIR" \
    --output-dir "output/test_clip" \
    --model-type clip \
    --clip-backbone ViT-B/16 \
    --batch-size 8 \
    --epochs 1 \
    --learning-rate 1e-4 \
    --n-splits 1 \
    --early-stopping-monitor val_mAP \
    --gpu-id 0 \
    2>&1 | head -50

if [ $? -eq 0 ]; then
    echo "✓ CLIP 模型测试通过"
else
    echo "✗ CLIP 模型测试失败"
fi

