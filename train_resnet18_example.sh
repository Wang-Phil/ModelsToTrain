#!/bin/bash
# ResNet18训练示例脚本
# 使用ImageNet预训练的ResNet18作为图像编码器

# 设置数据目录和输出目录
DATA_DIR="single_label_data"
OUTPUT_DIR="output/resnet18_imagenet_cv"

# 训练命令
python train_clip.py \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --image-encoder resnet18 \
    --text-encoder biomedclip_text \
    --embed-dim 512 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --temperature 0.07 \
    --img-size 224 \
    --augmentation standard \
    --use-cv \
    --n-splits 5 \
    --random-state 42 \
    --early-stopping-patience 10 \
    --use-superclip-loss \
    --class-loss-weight 1.0 \
    --contrastive-loss-weight 1.0 \
    --num-workers 4 \
    --gpu-id 0

echo "训练完成！结果保存在: ${OUTPUT_DIR}"

