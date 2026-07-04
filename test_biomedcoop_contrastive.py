#!/usr/bin/env python3
"""
测试 BiomedCoOp 对比损失模型是否可以正常训练
只训练图像编码器，只使用对比损失
"""

import sys
import os
import torch

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'BiomedCoOp'))
sys.path.insert(0, os.path.dirname(__file__))

def test_model_import():
    """测试模型是否可以正常导入"""
    print("=" * 60)
    print("测试模型导入...")
    print("=" * 60)
    
    models_to_test = [
        ('BiomedCoOp_PubMedCLIP', 'models.BiomedCoOp.biomedcoop_pubmedclip'),
        ('BiomedCoOp_CLIP', 'models.BiomedCoOp.biomedcoop_clip'),
        ('BiomedCoOp_PMCCLIP', 'models.BiomedCoOp.biomedcoop_pmcclip'),
        ('BiomedCoOp_BiomedCLIP', 'models.biomedcoop_biomedclip'),
    ]
    
    for trainer_name, module_path in models_to_test:
        try:
            print(f"\n测试 {trainer_name}...")
            module = __import__(module_path, fromlist=[trainer_name])
            trainer_class = getattr(module, trainer_name)
            print(f"  ✓ {trainer_name} 导入成功")
        except Exception as e:
            print(f"  ✗ {trainer_name} 导入失败: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("所有模型导入成功！")
    print("=" * 60)
    return True

def test_model_forward():
    """测试模型前向传播"""
    print("\n" + "=" * 60)
    print("测试模型前向传播（需要配置和数据集）...")
    print("=" * 60)
    print("注意：完整的前向传播测试需要配置文件和数据集")
    print("建议使用实际的训练脚本进行测试")
    print("=" * 60)
    return True

if __name__ == '__main__':
    print("BiomedCoOp 对比损失模型测试")
    print("=" * 60)
    
    # 测试导入
    if not test_model_import():
        print("\n❌ 模型导入测试失败！")
        sys.exit(1)
    
    # 测试前向传播（仅提示）
    test_model_forward()
    
    print("\n" + "=" * 60)
    print("✓ 基础测试通过！")
    print("=" * 60)
    print("\n下一步：使用训练脚本进行完整测试")
    print("示例命令：")
    print("  cd /home/ln/wangweicheng/BiomedCoOp")
    print("  bash scripts/biomedcoop/train_cv.sh \\")
    print("    data singlelabel BiomedCLIP")
    print("=" * 60)

