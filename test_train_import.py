"""
测试训练脚本的导入和基本功能
"""

import sys

def test_imports():
    """测试所有必要的导入"""
    print("测试导入...")
    
    try:
        import torch
        print("✓ torch")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        import torchvision
        print("✓ torchvision")
    except ImportError as e:
        print(f"✗ torchvision: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL")
    except ImportError as e:
        print(f"✗ PIL: {e}")
        return False
    
    try:
        from models.classic_models import create_model as create_classic_model
        print("✓ models.classic_models")
    except ImportError as e:
        print(f"✗ models.classic_models: {e}")
        return False
    
    try:
        from models.convnextv2 import convnextv2_tiny
        print("✓ models.convnextv2")
    except ImportError as e:
        print(f"✗ models.convnextv2: {e}")
        return False
    
    try:
        from models.starnext import starnext_tiny
        print("✓ models.starnext")
    except ImportError as e:
        print(f"✗ models.starnext: {e}")
        return False
    
    try:
        from models.starnet import starnet_s1
        print("✓ models.starnet")
    except ImportError as e:
        print(f"✗ models.starnet: {e}")
        return False
    
    try:
        from train_multiclass import create_model, get_data_augmentation, get_loss_function
        print("✓ train_multiclass")
    except ImportError as e:
        print(f"✗ train_multiclass: {e}")
        return False
    
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from train_multiclass import create_model
        
        # 测试经典模型
        models_to_test = [
            'resnet50',
            'convnextv2_tiny',
            'starnext_tiny',
            'starnet_s1',
        ]
        
        for model_name in models_to_test:
            try:
                model = create_model(model_name, num_classes=9, pretrained=False)
                num_params = sum(p.numel() for p in model.parameters())
                print(f"✓ {model_name}: {num_params:,} 参数")
            except Exception as e:
                print(f"✗ {model_name}: {e}")
        
        return True
    except Exception as e:
        print(f"✗ 模型创建测试失败: {e}")
        return False


def test_data_augmentation():
    """测试数据增强"""
    print("\n测试数据增强...")
    
    try:
        from train_multiclass import get_data_augmentation
        
        aug_types = ['none', 'minimal', 'standard', 'strong', 'medical']
        for aug_type in aug_types:
            try:
                train_transform, val_transform = get_data_augmentation(aug_type, img_size=224)
                print(f"✓ {aug_type}")
            except Exception as e:
                print(f"✗ {aug_type}: {e}")
        
        return True
    except Exception as e:
        print(f"✗ 数据增强测试失败: {e}")
        return False


def test_loss_functions():
    """测试损失函数"""
    print("\n测试损失函数...")
    
    try:
        from train_multiclass import get_loss_function
        
        loss_types = ['ce', 'focal', 'label_smoothing']
        for loss_type in loss_types:
            try:
                if loss_type == 'focal':
                    criterion = get_loss_function(loss_type, num_classes=9, focal_gamma=2.0)
                elif loss_type == 'label_smoothing':
                    criterion = get_loss_function(loss_type, num_classes=9, smoothing=0.1)
                else:
                    criterion = get_loss_function(loss_type, num_classes=9)
                print(f"✓ {loss_type}")
            except Exception as e:
                print(f"✗ {loss_type}: {e}")
        
        return True
    except Exception as e:
        print(f"✗ 损失函数测试失败: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("训练脚本功能测试")
    print("=" * 60)
    
    success = True
    success &= test_imports()
    success &= test_model_creation()
    success &= test_data_augmentation()
    success &= test_loss_functions()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("=" * 60)

