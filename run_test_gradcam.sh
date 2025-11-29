#!/bin/bash
# 快速测试CLIP热力图生成的脚本

# 设置参数
MODEL_PATH="checkpoints/clip_models/resnet18_clip:ViT-B/32/fold_1/checkpoint_best.pth"
IMAGE_ENCODER="resnet18"
TEXT_ENCODER="clip:ViT-B/32"
EMBED_DIM=512
CLASS_TEXTS_FILE="class_texts_hip_prosthesis.json"
OUTPUT_DIR="clip_gradcam_results"

# 测试图片列表
TEST_IMAGES=(
    "data/test/Good Place/263_l.jpg"
    "data/test/Native Hip/346_r.jpg"
    "data/test/Acetabular Loosening/1012_r.jpg"
)

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "CLIP模型热力图生成测试"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "图像编码器: $IMAGE_ENCODER"
echo "文本编码器: $TEXT_ENCODER"
echo "输出目录: $OUTPUT_DIR"
echo "测试图片数: ${#TEST_IMAGES[@]}"
echo "=========================================="
echo ""

# 处理每张图片
for img_path in "${TEST_IMAGES[@]}"; do
    if [ -f "$img_path" ]; then
        echo "处理图片: $img_path"
        
        img_name=$(basename "$img_path" .jpg)
        output_name="gradcam_${img_name}.png"
        output_path="$OUTPUT_DIR/$output_name"
        
        python clip_gradcam_visualization.py \
            --model-path "$MODEL_PATH" \
            --image-path "$img_path" \
            --class-texts-file "$CLASS_TEXTS_FILE" \
            --image-encoder "$IMAGE_ENCODER" \
            --text-encoder "$TEXT_ENCODER" \
            --embed-dim $EMBED_DIM \
            --output-dir "$OUTPUT_DIR" \
            --output-name "$output_name"
        
        if [ $? -eq 0 ]; then
            echo "✓ 成功生成: $output_path"
        else
            echo "✗ 生成失败: $img_path"
        fi
        echo ""
    else
        echo "✗ 图片不存在: $img_path"
    fi
done

echo "=========================================="
echo "测试完成！结果保存在: $OUTPUT_DIR"
echo "=========================================="

