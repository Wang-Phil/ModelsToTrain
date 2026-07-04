#!/bin/bash
# CLIP 训练：使用 agent_run_v2 生成的短词（class_texts_hip_prosthesis_agent_v2.json）
# 与 ablation_clip_short_doctor_text 区分：不同 class_texts、不同输出目录
# 用法: bash run_train_clip_agent_v2.sh [gpu_id]

set -e
cd "$(dirname "$0")"
GPU_ID=${1:-0}

python train_clip.py \
  --data-dir single_label_data \
  --output-dir checkpoints/clip_agent_v2/resnet18_clip_ViT-B_32 \
  --config-file config_train_clip_agent_v2.json \
  --multi-config \
  --gpu-id "$GPU_ID"
