#!/usr/bin/env python
"""
测试CLIP ResNet功能
验证修改后的代码是否能正常工作
"""

import sys
import torch
sys.path.insert(0, 'models')
sys.path.insert(0, '.')

def test_imagenet_resnet18():
    """测试ImageNet预训练的ResNet18（不依赖CLIP）"""
    print("\n" + "="*60)
    print("测试1: ImageNet预训练的ResNet18")
    print("="*60)
    
    try:
        from models.clip import ImageEncoder
        
        encoder = ImageEncoder(model_name='resnet18', embed_dim=512)
        print('✓ ResNet18加载成功')
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f'✓ 前向传播成功，输出形状: {output.shape}')
        
        if output.shape == (2, 512):
            print('✓ 输出维度正确')
            print('✓ ResNet18可以正常训练（使用ImageNet预训练权重）')
            return True
        else:
            print(f'✗ 输出维度错误，期望 [2, 512]，实际 {output.shape}')
            return False
            
    except Exception as e:
        print(f'✗ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_clip_resnet50():
    """测试CLIP预训练的ResNet50"""
    print("\n" + "="*60)
    print("测试2: CLIP预训练的ResNet50")
    print("="*60)
    
    # 先检查CLIP是否可用
    try:
        import clip
        if not hasattr(clip, 'load'):
            print('✗ CLIP库不可用（缺少load函数）')
            print('  请安装: pip install git+https://github.com/openai/CLIP.git')
            return False
    except ImportError:
        print('✗ CLIP库未安装')
        print('  请安装: pip install git+https://github.com/openai/CLIP.git')
        return False
    
    try:
        from models.clip import ImageEncoder
        
        print('加载CLIP ResNet50...')
        encoder = ImageEncoder(model_name='resnet50:clip', embed_dim=512)
        print('✓ CLIP ResNet50加载成功')
        
        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = encoder(dummy_input)
        
        print(f'✓ 前向传播成功，输出形状: {output.shape}')
        
        if output.shape == (2, 512):
            print('✓ 输出维度正确')
            print('✓ CLIP ResNet50可以正常训练')
            return True
        else:
            print(f'✗ 输出维度错误，期望 [2, 512]，实际 {output.shape}')
            return False
            
    except Exception as e:
        print(f'✗ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

def test_clip_model():
    """测试完整的CLIP模型"""
    print("\n" + "="*60)
    print("测试3: 完整的CLIP模型（ResNet18 + BERT）")
    print("="*60)
    
    try:
        from models.clip import CLIPModel
        
        print('创建CLIP模型...')
        model = CLIPModel(
            image_encoder_name='resnet18',
            text_encoder_name='bert-base-chinese',
            embed_dim=512,
            temperature=0.07
        )
        print('✓ CLIP模型创建成功')
        
        # 测试图像编码
        dummy_images = torch.randn(2, 3, 224, 224)
        dummy_texts = ['测试文本1', '测试文本2']
        
        with torch.no_grad():
            image_features, text_features = model(dummy_images, texts=dummy_texts)
        
        print(f'✓ 前向传播成功')
        print(f'  图像特征形状: {image_features.shape}')
        print(f'  文本特征形状: {text_features.shape}')
        
        if image_features.shape == (2, 512) and text_features.shape == (2, 512):
            print('✓ 特征维度正确')
            print('✓ 完整CLIP模型可以正常训练')
            return True
        else:
            print(f'✗ 特征维度错误')
            return False
            
    except Exception as e:
        print(f'✗ 测试失败: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("CLIP ResNet功能测试")
    print("="*60)
    
    results = []
    
    # 测试1: ImageNet ResNet18（不依赖CLIP）
    results.append(('ImageNet ResNet18', test_imagenet_resnet18()))
    
    # 测试2: CLIP ResNet50（需要CLIP库）
    results.append(('CLIP ResNet50', test_clip_resnet50()))
    
    # 测试3: 完整CLIP模型
    results.append(('完整CLIP模型', test_clip_model()))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:30s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！可以正常训练")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("="*60)
    
    # 使用建议
    print("\n使用建议:")
    print("1. ResNet18: 使用 --image-encoder resnet18 (ImageNet预训练)")
    print("2. CLIP ResNet50: 使用 --image-encoder resnet50:clip (需要安装CLIP库)")
    print("3. CLIP ResNet101: 使用 --image-encoder resnet101:clip (需要安装CLIP库)")
    print("\n注意: CLIP不提供ResNet18，如需ResNet18请使用ImageNet预训练版本")

