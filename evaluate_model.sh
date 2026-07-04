#!/bin/bash

# 评估增强版CLIP模型的mAP指标
# 模型路径: outputs/enhanced_clip_full_singlelabel_resnet50pmcclip_biomedclip_gpu4/best_model.pth

python evaluate_clip_enhanced.py \
    --model-path ./outputs/enhanced_clip_full_singlelabel_resnet50pmcclip_biomedclip_gpu4/best_model.pth \
    --data-dir ./single_label_data \
    --image-encoder resnet50:pmcclip \
    --text-encoder biomedclip_text \
    --embed-dim 512 \
    --temperature 0.07 \
    --class-texts-file class_texts_hip_prosthesis.json \
    --img-size 224 \
    --batch-size 32 \
    --num-workers 4 \
    --gpu-id 5 \

