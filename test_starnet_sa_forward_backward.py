"""
测试 StarNet SA 变体模型的前向传播和反向传播
"""

import torch
import torch.nn as nn
import sys
import os

# 添加模型路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from starnet_sa_variants import (
    starnet_sa_s1,
    starnet_sa_s2,
    starnet_sa_s3,
    starnet_sa_s4,
    StarNet_SA
)

def test_model_forward_backward(model, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    测试模型的前向传播和反向传播
    
    Args:
        model: 模型实例
        model_name: 模型名称
        device: 设备 ('cuda' 或 'cpu')
    """
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name}")
    print(f"设备: {device}")
    print(f"{'='*60}")
    
    model = model.to(device)
    model.train()  # 设置为训练模式以启用dropout等
    
    # 创建测试输入
    batch_size = 2
    input_size = 224
    x = torch.randn(batch_size, 3, input_size, input_size).to(device)
    
    try:
        # 1. 测试前向传播
        print("\n[1] 测试前向传播...")
        with torch.no_grad():
            output = model(x)
        print(f"    ✓ 前向传播成功!")
        print(f"    输入形状: {x.shape}")
        print(f"    输出形状: {output.shape}")
        print(f"    输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 检查输出是否包含 NaN 或 Inf
        if torch.isnan(output).any():
            print(f"    ✗ 警告: 输出包含 NaN!")
            return False
        if torch.isinf(output).any():
            print(f"    ✗ 警告: 输出包含 Inf!")
            return False
        
        # 2. 测试反向传播
        print("\n[2] 测试反向传播...")
        model.zero_grad()
        output = model(x)  # 重新前向传播（不使用no_grad）
        
        # 创建虚拟损失
        target = torch.randint(0, model.num_classes, (batch_size,)).to(device)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        print(f"    Loss值: {loss.item():.4f}")
        
        # 反向传播
        loss.backward()
        print(f"    ✓ 反向传播成功!")
        
        # 检查梯度
        has_grad = False
        no_grad_count = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                has_grad = True
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any():
                    print(f"    ✗ 警告: {name} 的梯度包含 NaN!")
                    return False
                if torch.isinf(param.grad).any():
                    print(f"    ✗ 警告: {name} 的梯度包含 Inf!")
                    return False
            else:
                no_grad_count += 1
        
        print(f"    总参数数量: {total_params}")
        print(f"    有梯度的参数: {total_params - no_grad_count}")
        print(f"    无梯度的参数: {no_grad_count}")
        
        if not has_grad:
            print(f"    ✗ 警告: 没有任何参数有梯度!")
            return False
        
        # 3. 测试不同输入尺寸
        print("\n[3] 测试不同输入尺寸...")
        test_sizes = [(1, 3, 224, 224), (4, 3, 256, 256), (2, 3, 112, 112)]
        for size in test_sizes:
            try:
                x_test = torch.randn(*size).to(device)
                with torch.no_grad():
                    output_test = model(x_test)
                print(f"    ✓ 输入尺寸 {size} -> 输出形状 {output_test.shape}")
            except Exception as e:
                print(f"    ✗ 输入尺寸 {size} 失败: {e}")
                return False
        
        print(f"\n✓ {model_name} 测试通过!")
        return True
        
    except Exception as e:
        print(f"\n✗ {model_name} 测试失败!")
        print(f"   错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("="*60)
    print("StarNet SA 变体模型前向/反向传播测试")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    
    # 测试所有模型变体
    models_to_test = [
        (starnet_sa_s1, "starnet_sa_s1"),
        (starnet_sa_s2, "starnet_sa_s2"),
        (starnet_sa_s3, "starnet_sa_s3"),
        (starnet_sa_s4, "starnet_sa_s4"),
    ]
    
    results = {}
    for model_fn, model_name in models_to_test:
        try:
            model = model_fn(pretrained=False, num_classes=1000)
            success = test_model_forward_backward(model, model_name, device)
            results[model_name] = success
        except Exception as e:
            print(f"\n✗ {model_name} 实例化失败: {e}")
            results[model_name] = False
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for model_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{model_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有模型测试通过!")
    else:
        print("\n⚠️  部分模型测试失败，请检查上述错误信息")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

