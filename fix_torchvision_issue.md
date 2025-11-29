# 修复 torch._C.Tag 错误

## 问题
`AttributeError: module 'torch._C' has no attribute 'Tag'` 
这是 PyTorch 和 torchvision 版本不匹配导致的。

## 解决方案

### 方案 1: 升级 PyTorch（推荐）
```bash
# 在 wangweicheng 环境中
conda activate wangweicheng
pip install --upgrade torch torchvision
```

### 方案 2: 降级 torchvision
```bash
conda activate wangweicheng
pip install torchvision==0.15.2  # 或其他兼容版本
```

### 方案 3: 检查并修复版本匹配
```bash
conda activate wangweicheng
# 检查当前版本
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"

# 根据 PyTorch 版本安装匹配的 torchvision
# PyTorch 2.0+ 需要 torchvision 0.15+
# PyTorch 2.1+ 需要 torchvision 0.16+
# PyTorch 2.2+ 需要 torchvision 0.17+
```

## 验证
```bash
python -c "import torch; import torchvision; print('✓ 导入成功')"
```
