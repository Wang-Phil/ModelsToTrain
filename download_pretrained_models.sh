#!/bin/bash

# 下载 CoOp 模型所需的预训练模型

echo "============================================================"
echo "下载 CoOp 模型预训练权重"
echo "============================================================"
echo ""

# 检测网络连接并选择镜像站点
check_network() {
    if curl -s --connect-timeout 3 https://huggingface.co > /dev/null 2>&1; then
        echo "✓ 可以连接到 huggingface.co，使用官方站点"
        echo "BASE_URL=https://huggingface.co"
        return 0
    else
        echo "⚠ 无法连接到 huggingface.co，使用镜像站点: hf-mirror.com"
        echo "BASE_URL=https://hf-mirror.com"
        return 1
    fi
}

# 检测网络并设置基础 URL
if check_network; then
    BASE_URL="https://huggingface.co"
else
    BASE_URL="https://hf-mirror.com"
fi

# 创建检查点目录
CHECKPOINT_DIR="clip/checkpoints"
mkdir -p ${CHECKPOINT_DIR}

# 1. CoOp_CLIP (标准 CLIP)
echo "【1. CoOp_CLIP (标准 CLIP)】"
echo "----------------------------------------"
echo "模型: 标准 CLIP (ViT-B/16, ViT-B/32, RN50 等)"
echo "下载方式: 自动下载（首次使用时）"
echo "下载位置: CLIP 库的默认缓存目录"
echo "说明: 运行训练时会自动下载，无需手动下载"
echo ""

# 2. CoOp_PMCCLIP
echo "【2. CoOp_PMCCLIP】"
echo "----------------------------------------"
echo "需要下载的文件（保存到 ${CHECKPOINT_DIR}/）:"
echo "  1. text_encoder.pth"
echo "  2. image_encoder(resnet50).pth"
echo "  3. text_projection_layer.pth"
echo ""

FILES_PMC=(
    "text_encoder.pth|https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth"
    "image_encoder(resnet50).pth|https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth"
    "text_projection_layer.pth|https://huggingface.co/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth"
)

echo "开始下载 PMC-CLIP 模型文件..."
for file_info in "${FILES_PMC[@]}"; do
    IFS='|' read -r filename url <<< "$file_info"
    filepath="${CHECKPOINT_DIR}/${filename}"
    
    if [ -f "${filepath}" ]; then
        echo "✓ ${filename} 已存在，跳过下载"
    else
        echo "下载 ${filename}..."
        wget -O "${filepath}" "${url}" || curl -L -o "${filepath}" "${url}"
        if [ $? -eq 0 ]; then
            echo "✓ ${filename} 下载完成"
        else
            echo "✗ ${filename} 下载失败"
        fi
    fi
done
echo ""

# 3. CoOp_PubMedCLIP
echo "【3. CoOp_PubMedCLIP】"
echo "----------------------------------------"
echo "需要下载的文件（保存到 ${CHECKPOINT_DIR}/）:"
echo "  1. PubMedCLIP_ViT32.pth"
echo "  2. 标准 CLIP ViT-B/32（自动下载）"
echo ""

FILES_PUBMED=(
    "PubMedCLIP_ViT32.pth|https://huggingface.co/sarahESL/PubMedCLIP/resolve/main/PubMedCLIP_ViT32.pth?download=true"
)

echo "开始下载 PubMedCLIP 模型文件..."
for file_info in "${FILES_PUBMED[@]}"; do
    IFS='|' read -r filename url <<< "$file_info"
    filepath="${CHECKPOINT_DIR}/${filename}"
    
    if [ -f "${filepath}" ]; then
        echo "✓ ${filename} 已存在，跳过下载"
    else
        echo "下载 ${filename}..."
        wget -O "${filepath}" "${url}" || curl -L -o "${filepath}" "${url}"
        if [ $? -eq 0 ]; then
            echo "✓ ${filename} 下载完成"
        else
            echo "✗ ${filename} 下载失败"
        fi
    fi
done
echo ""

echo "============================================================"
echo "下载完成！"
echo "============================================================"
echo "文件位置: ${CHECKPOINT_DIR}/"
echo ""

