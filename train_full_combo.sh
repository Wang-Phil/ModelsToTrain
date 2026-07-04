#!/bin/bash

# 完整组合训练：所有损失函数（分类损失 + 对比损失 + SCCM损失 + KDSP损失）
# 数据：single_label_data (9个类别)
# GPU: 第5个 (gpu-id 4)

python train_clip_enhanced.py \
    --data-dir ./single_label_data \
    --output-dir ./outputs/enhanced_clip_full_singlelabel_resnet50pmcclip_biomedclip_gpu4 \
    --image-encoder resnet50:pmcclip \
    --text-encoder biomedclip_text \
    --embed-dim 512 \
    --temperature 0.07 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --weight-decay 0.01 \
    --img-size 224 \
    --augmentation standard \
    --class-texts-file class_texts_hip_prosthesis.json \
    --teacher-image-encoder biomedclip \
    --teacher-text-encoder biomedclip_text \
    --use-classification-loss \
    --use-contrastive-loss \
    --use-sccm-loss \
    --use-kdsp-loss \
    --classification-loss-weight 0.5 \
    --contrastive-loss-weight 0.5 \
    --sccm-loss-weight 1.0 \
    --kdsp-loss-weight 1.0 \
    --num-workers 4 \
    --gpu-id 5

