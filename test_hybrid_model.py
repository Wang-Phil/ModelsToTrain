#!/usr/bin/env python3
"""
测试混合模型（PMC-CLIP ResNet50 + BiomedCLIP 文本编码器）
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_test_config():
    """创建测试配置"""
    class CfgNode(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def __setattr__(self, name, value):
            self[name] = value

    cfg = CfgNode()

    # 基本配置
    cfg.INPUT = CfgNode()
    cfg.INPUT.SIZE = [224, 224]

    cfg.OPTIM = CfgNode()
    cfg.OPTIM.MAX_EPOCH = 10

    # BiomedCoOp 配置
    cfg.TRAINER = CfgNode()
    cfg.TRAINER.BIOMEDCOOP = CfgNode()
    cfg.TRAINER.BIOMEDCOOP.N_CTX = 4
    cfg.TRAINER.BIOMEDCOOP.CTX_INIT = "a photo of a"
    cfg.TRAINER.BIOMEDCOOP.CSC = False
    cfg.TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION = "end"
    cfg.TRAINER.BIOMEDCOOP.SCCM_LAMBDA = 1.0
    cfg.TRAINER.BIOMEDCOOP.USE_FOCAL_LOSS = False
    cfg.TRAINER.BIOMEDCOOP.FOCAL_ALPHA = 0.25
    cfg.TRAINER.BIOMEDCOOP.FOCAL_GAMMA = 2.0
    cfg.TRAINER.BIOMEDCOOP.CLASS_TEXTS_FILE = None
    cfg.TRAINER.BIOMEDCOOP.PREC = "fp32"
    cfg.TRAINER.BIOMEDCOOP.CLASSIFICATION_LOSS_WEIGHT = 0.5
    cfg.TRAINER.BIOMEDCOOP.CONTRASTIVE_LOSS_WEIGHT = 0.5

    cfg.MODEL = CfgNode()
    cfg.MODEL.INIT_WEIGHTS = None

    return cfg

def test_hybrid_model():
    """测试混合模型"""
    print("=" * 80)
    print("测试混合模型：PMC-CLIP ResNet50 + BiomedCLIP 文本编码器")
    print("=" * 80)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建配置
    cfg = create_test_config()
    classnames = ['class1', 'class2', 'class3']  # 测试类别

    try:
        # 导入混合模型
        from models.hybrid_pmcclip_biomedclip import HybridCLIP
        print("✓ 成功导入混合模型类")

        # 创建模型
        print("\n创建混合模型...")
        model = HybridCLIP(cfg, classnames, None, None)  # 先用None测试
        print("✓ 混合模型创建成功")

        # 测试模型参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数: {total_params:,}")

        # 测试前向传播（模拟数据）
        batch_size = 4
        image_size = 224

        # 创建模拟图像数据
        images = torch.randn(batch_size, 3, image_size, image_size).to(device)
        labels = torch.randint(0, len(classnames), (batch_size,)).to(device)

        print(f"\n测试前向传播...")
        print(f"  输入图像尺寸: {images.shape}")
        print(f"  标签: {labels}")

        # 训练模式
        model.train()
        logits, loss_ce, contrastive_loss, loss_sccm = model(images, labels)

        print("✓ 训练模式前向传播成功"        print(f"  Logits 尺寸: {logits.shape}")
        print(f"  分类损失: {loss_ce.item():.4f}")
        print(f"  对比损失: {contrastive_loss.item():.4f}")
        print(f"  SCCM损失: {loss_sccm.item():.4f}")

        # 评估模式
        model.eval()
        with torch.no_grad():
            eval_logits = model(images)
            print("✓ 评估模式前向传播成功"            print(f"  评估Logits 尺寸: {eval_logits.shape}")

        print("\n" + "=" * 80)
        print("✅ 混合模型测试通过！")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_hybrid_model()
    sys.exit(0 if success else 1)
