#!/bin/bash
# ResNet18 + LSAL 训练脚本
# 注意：需要先修改 lsal_biomedclip.py 以支持ResNet18

set -e

# 设置环境变量
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0  # 设置使用的GPU ID

# ============================================
# 配置区域
# ============================================

# 数据目录
DATA_DIR="single_label_data"

# 输出目录
OUTPUT_DIR="output/lsal_resnet18"

# 配置文件路径（需要先创建）
CONFIG_FILE="configs/trainers/LSAL_BiomedCLIP/resnet18.yaml"

# 语义文件目录
SEMANTICS_DIR="semantics"

# 随机种子
SEED=42

# ============================================

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}ResNet18 + LSAL 训练${NC}"
echo -e "${YELLOW}========================================${NC}"

# 检查语义文件是否存在
if [ ! -f "${SEMANTICS_DIR}/class_centers.pt" ] || [ ! -f "${SEMANTICS_DIR}/soft_labels_matrix.pt" ]; then
    echo -e "${RED}错误: 语义文件不存在！${NC}"
    echo -e "${YELLOW}请先运行以下命令生成语义文件：${NC}"
    echo "python models/build_llm_semantics.py \\"
    echo "    --classnames-file ${DATA_DIR}/classnames.json \\"
    echo "    --output-dir ${SEMANTICS_DIR} \\"
    echo "    --tau 0.1"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo -e "${RED}错误: 配置文件不存在！${NC}"
    echo -e "${YELLOW}请先创建配置文件: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}参考: configs/trainers/LSAL_BiomedCLIP/vit_b16.yaml${NC}"
    exit 1
fi

# 检查lsal_biomedclip.py是否已修改
echo -e "${YELLOW}注意: 需要先修改 models/lsal_biomedclip.py 以支持ResNet18${NC}"
echo -e "${YELLOW}参考: models/lsal_resnet18_example.py${NC}"
read -p "是否已修改代码？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${RED}请先修改代码后再运行此脚本${NC}"
    exit 1
fi

# 进入Dassl目录
cd Dassl.pytorch

# 运行训练
echo -e "${GREEN}开始训练...${NC}"
python tools/train.py \
    --root "../${DATA_DIR}" \
    --trainer LSAL_BiomedCLIP \
    --config-file "../${CONFIG_FILE}" \
    --output-dir "../${OUTPUT_DIR}" \
    --seed ${SEED}

echo -e "${GREEN}训练完成！结果保存在: ${OUTPUT_DIR}${NC}"

