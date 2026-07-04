#!/bin/bash

# 重启 Grad-CAM 生成，使用新的单图版本

echo "=========================================="
echo "重启 Grad-CAM 生成（单图版本）"
echo "=========================================="

# 停止旧进程
echo "停止旧进程..."
pkill -f "generate_cv_gradcam.py"
sleep 2

# 检查是否停止成功
if ps aux | grep -q "[g]enerate_cv_gradcam"; then
    echo "警告: 仍有进程在运行，强制停止..."
    pkill -9 -f "generate_cv_gradcam.py"
    sleep 1
fi

echo "✓ 旧进程已停止"
echo ""

# 可选：清理已生成的文件（如果需要重新生成）
read -p "是否删除已生成的旧图片（三图版本）？[y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "删除旧图片..."
    rm -rf checkpoints/final_models/sk_size_ablation/starnet_s1_sk39/gradcam_output/fold_*/class_*/*.png
    echo "✓ 旧图片已删除"
else
    echo "保留旧图片"
fi

echo ""

# 启动新进程（生成单图版本）
echo "启动新进程（单图版本）..."
nohup python generate_cv_gradcam.py \
    --model-dir checkpoints/final_models/sk_size_ablation/starnet_s1_sk39 \
    --data-dir single_label_data \
    --folds all \
    --device cuda:0 \
    > gradcam_single_image.log 2>&1 &

echo "✓ 新进程已启动（PID: $!）"
echo ""
echo "输出日志: gradcam_single_image.log"
echo "输出目录: checkpoints/final_models/sk_size_ablation/starnet_s1_sk39/gradcam_output/"
echo ""
echo "监控命令: tail -f gradcam_single_image.log"
echo "=========================================="

