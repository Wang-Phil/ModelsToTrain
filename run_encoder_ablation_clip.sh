#!/usr/bin/bash
#
# 图像编码器消融：固定文本编码器 clip:ViT-B/32；训练仅 CLIP 对称对比损失（无 SupCon/分类/Focal）；
# 不使用加权采样；其余超参与 train_clip_config_best_run.json 对齐（见
# train_clip_config_encoder_ablation_clip_only.json）。
# 仅替换 image_encoder_name。
#
# 对比项：
#   1) resnet18              — ImageNet 预训练 ResNet18
#   2) resnet50:clip         — OpenAI CLIP RN50 visual
#   3) resnet50              — ImageNet 预训练 ResNet50
#   4) starnet_s1:pretrained — StarNet S1 + 官方预训练
#   5) casgnet               — CasGNet（默认骨干见 models/casgnet.py，可换结构/权重）
#
# CasGNet 权重（任选其一）：
#   - 配置名写成 casgnet:/绝对路径/weights.pth
#   - 或 export CASGNET_WEIGHTS=/path/to.pth 且使用 casgnet:pretrained
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE_CONFIG="${BASE_CONFIG:-train_clip_config_encoder_ablation_clip_only.json}"
DATA_DIR="${DATA_DIR:-single_label_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/encoder_ablation_clip}"
GPU_ID="${GPU_ID:-9}"

TEXT_ENCODER_FIXED="${TEXT_ENCODER_FIXED:-clip:ViT-B/32}"

if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "未找到基础配置: $BASE_CONFIG"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "未找到数据目录: $DATA_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

ENCODERS=(
  "resnet18"
  "resnet50:clip"
  "resnet50"
  "starnet_s1:pretrained"
  "casgnet"
)

echo "============================================================"
echo "编码器消融 | 仅 CLIP loss | 无加权采样 | 固定文本: $TEXT_ENCODER_FIXED"
echo "基础配置: $BASE_CONFIG"
echo "数据目录: $DATA_DIR"
echo "输出根目录: $OUTPUT_ROOT"
echo "GPU_ID: $GPU_ID"
echo "============================================================"

for ENC in "${ENCODERS[@]}"; do
  name_safe="${ENC//:/_}"
  name_safe="${name_safe//\//_}"
  OUT="${OUTPUT_ROOT}/${name_safe}"
  TMP_CFG="$(mktemp /tmp/clip_enc_XXXXXX.json)"

  python3 - "$BASE_CONFIG" "$ENC" "$TEXT_ENCODER_FIXED" "$GPU_ID" "$TMP_CFG" <<'PY'
import json
import sys

base_path, enc, text_enc, gpu_s, out_path = sys.argv[1:6]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)[0]
cfg["image_encoder_name"] = enc
cfg["text_encoder_name"] = text_enc
cfg["gpu_id"] = int(gpu_s)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump([cfg], f, indent=2, ensure_ascii=False)
PY

  echo ""
  echo ">>> 运行: image_encoder_name=$ENC -> $OUT"
  python train_clip.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT" \
    --config-file "$TMP_CFG" \
    --multi-config \
    --gpu-id "$GPU_ID"

  rm -f "$TMP_CFG"
done

echo ""
echo "全部完成。各实验 cv_summary.json 位于: $OUTPUT_ROOT/*/cv_summary.json"
