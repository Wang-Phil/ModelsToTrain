"""
测试 ResNet18 预训练权重加载
用于验证预训练权重是否可以正常下载和加载
"""

import torch
from models.classic_models import get_resnet18

print("=" * 60)
print("测试 ResNet18 预训练权重加载")
print("=" * 60)
print()

# 测试不使用预训练权重
print("1. 测试不使用预训练权重（随机初始化）...")
try:
    model_no_pretrained = get_resnet18(num_classes=9, pretrained=False)
    print("✓ 成功创建模型（不使用预训练权重）")
    print(f"  参数量: {sum(p.numel() for p in model_no_pretrained.parameters()):,}")
except Exception as e:
    print(f"✗ 失败: {e}")
    exit(1)

print()

# 测试使用预训练权重
print("2. 测试使用预训练权重（自动下载）...")
print("   注意: 如果本地没有权重，PyTorch 会自动从网络下载")
print("   权重会保存在: ~/.cache/torch/hub/checkpoints/")
print()

try:
    model_pretrained = get_resnet18(num_classes=9, pretrained=True)
    print("✓ 成功创建模型（使用预训练权重）")
    print(f"  参数量: {sum(p.numel() for p in model_pretrained.parameters()):,}")
    
    # 测试前向传播
    print()
    print("3. 测试前向传播...")
    model_pretrained.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model_pretrained(dummy_input)
    print(f"✓ 前向传播成功")
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    print()
    print("=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print()
    print("提示:")
    print("  - PyTorch 会自动下载预训练权重（如果本地没有）")
    print("  - 权重保存在: ~/.cache/torch/hub/checkpoints/")
    print("  - 下载一次后，后续训练会直接使用本地缓存")
    print("  - 如果网络不好，可以手动下载权重文件")
    print()
    
except Exception as e:
    print(f"✗ 失败: {e}")
    print()
    print("可能的原因:")
    print("  1. 网络连接问题（无法下载权重）")
    print("  2. PyTorch 版本过低（需要 1.13+）")
    print("  3. 磁盘空间不足")
    print()
    print("解决方案:")
    print("  1. 检查网络连接")
    print("  2. 升级 PyTorch: pip install --upgrade torch torchvision")
    print("  3. 手动下载权重（见下方说明）")
    exit(1)






