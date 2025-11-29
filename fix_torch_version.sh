#!/bin/bash
# 修复 PyTorch 和 torchvision 版本不匹配问题

echo "=== 当前版本 ==="
source /home/ln/anaconda3/etc/profile.d/conda.sh
conda activate wangweicheng
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)" 2>&1 | head -1

echo ""
echo "=== 解决方案 ==="
echo ""
echo "方案 1: 升级 PyTorch 到 2.0+（推荐，兼容 DCNv4）"
echo "  conda activate wangweicheng"
echo "  pip install torch>=2.0.0 torchvision>=0.15.0 --upgrade"
echo ""
echo "方案 2: 降级 torchvision 到与 PyTorch 1.12.1 兼容的版本"
echo "  conda activate wangweicheng"
echo "  pip install torchvision==0.13.1"
echo ""
echo "推荐使用方案 1，因为 DCNv4 可能需要较新的 PyTorch 版本"
