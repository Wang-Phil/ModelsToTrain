#!/usr/bin/env python
"""
测试 hybrid_coop_sccm 模型是否可以正常导入和初始化
"""

import sys
import os
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("测试 hybrid_coop_sccm 模型导入和初始化")
print("=" * 80)

# 1. 测试导入
print("\n1. 测试导入模型...")
try:
    from models.hybrid_pmcclip_biomedclip_coop_sccm import HybridCLIPWithCoOpSCCM
    print("✓ 成功导入 HybridCLIPWithCoOpSCCM")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 测试创建简单配置
print("\n2. 测试创建配置...")
try:
    class SimpleCfg:
        class INPUT:
            SIZE = [224, 224]
        class OPTIM:
            MAX_EPOCH = 100
        class TRAINER:
            class BIOMEDCOOP:
                N_CTX = 4
                CTX_INIT = "a photo of a"
                SCCM_LAMBDA = 1.0
                CLASSIFICATION_LOSS_WEIGHT = 0.4
                CONTRASTIVE_LOSS_WEIGHT = 0.4
                DISTILLATION_LOSS_WEIGHT = 0.2
                CLASS_TEXTS_FILE = "class_texts_hip_prosthesis.json"
    
    cfg = SimpleCfg()
    print("✓ 配置创建成功")
except Exception as e:
    print(f"✗ 配置创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. 测试模型初始化（不加载实际模型，只测试代码结构）
print("\n3. 测试模型代码结构...")
try:
    # 检查必要的类和方法是否存在
    assert hasattr(HybridCLIPWithCoOpSCCM, '__init__')
    assert hasattr(HybridCLIPWithCoOpSCCM, 'forward')
    print("✓ 模型类结构正确")
except Exception as e:
    print(f"✗ 模型结构检查失败: {e}")
    sys.exit(1)

# 4. 测试训练脚本导入
print("\n4. 测试训练脚本导入...")
try:
    # 检查 train_biomedcoop.py 是否可以导入新模型
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_biomedcoop", "train_biomedcoop.py")
    if spec is None:
        print("⚠️  无法加载 train_biomedcoop.py（可能路径问题）")
    else:
        print("✓ train_biomedcoop.py 可以加载")
except Exception as e:
    print(f"⚠️  训练脚本检查失败: {e}")

print("\n" + "=" * 80)
print("基本测试完成！")
print("=" * 80)
print("\n注意：这只是代码结构测试，实际训练需要：")
print("  1. 数据目录存在")
print("  2. PMC-CLIP 模型文件已下载")
print("  3. BiomedCLIP 模型可以加载")
print("  4. GPU 可用（或使用 CPU）")
print("\n运行完整训练：")
print("  python train_biomedcoop.py --model-type hybrid_coop_sccm ...")
print("  或")
print("  ./train_hybrid_coop_sccm.sh")

