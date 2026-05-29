#!/bin/bash

# 增强版CLIP模型训练示例脚本
# 根据你的需求修改参数

# ========== 示例1: 基础训练（分类损失 + 对比损失）==========
echo "开始基础训练..."
python train_clip_enhanced.py \
    --data-dir ./data/hip_prosthesis \
    --output-dir ./outputs/basic_training \
    --image-encoder resnet50 \
    --text-encoder pubmedbert \
    --embed-dim 512 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --img-size 224 \
    --augmentation standard \
    --classification-loss-weight 1.0 \
    --contrastive-loss-weight 1.0 \
    --gpu-id 0

# ========== 示例2: 使用SCCM损失（需要类别文本描述文件）==========
# echo "开始SCCM训练..."
# python train_clip_enhanced.py \
#     --data-dir ./data/hip_prosthesis \
#     --output-dir ./outputs/sccm_training \
#     --image-encoder resnet50 \
#     --text-encoder biomedclip_text \
#     --embed-dim 512 \
#     --batch-size 32 \
#     --epochs 100 \
#     --learning-rate 1e-4 \
#     --class-texts-file class_texts_hip_prosthesis.json \
#     --use-sccm-loss \
#     --classification-loss-weight 0.5 \
#     --contrastive-loss-weight 0.5 \
#     --sccm-loss-weight 1.0 \
#     --gpu-id 0

# ========== 示例3: 使用KDSP损失（需要teacher模型）==========
# echo "开始KDSP训练..."
# python train_clip_enhanced.py \
#     --data-dir ./data/hip_prosthesis \
#     --output-dir ./outputs/kdsp_training \
#     --image-encoder resnet50 \
#     --text-encoder pubmedbert \
#     --embed-dim 512 \
#     --batch-size 32 \
#     --epochs 100 \
#     --learning-rate 1e-4 \
#     --teacher-image-encoder biomedclip \
#     --teacher-text-encoder biomedclip_text \
#     --use-kdsp-loss \
#     --classification-loss-weight 0.5 \
#     --contrastive-loss-weight 0.5 \
#     --kdsp-loss-weight 1.0 \
#     --gpu-id 0

# ========== 示例4: 完整组合（所有损失函数）==========
# echo "开始完整组合训练..."
# python train_clip_enhanced.py \
#     --data-dir ./data/hip_prosthesis \
#     --output-dir ./outputs/full_training \
#     --image-encoder resnet50:pmcclip \
#     --text-encoder biomedclip_text \
#     --embed-dim 512 \
#     --batch-size 32 \
#     --epochs 100 \
#     --learning-rate 1e-4 \
#     --class-texts-file class_texts_hip_prosthesis.json \
#     --teacher-image-encoder biomedclip \
#     --teacher-text-encoder biomedclip_text \
#     --use-sccm-loss \
#     --use-kdsp-loss \
#     --classification-loss-weight 0.5 \
#     --contrastive-loss-weight 0.5 \
#     --sccm-loss-weight 1.0 \
#     --kdsp-loss-weight 1.0 \
#     --gpu-id 0

echo "训练完成！"

