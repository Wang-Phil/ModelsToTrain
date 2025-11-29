"""
测试CLIP模型是否能正常初始化和前向传播
"""

import torch
import sys
from pathlib import Path

# 添加模型路径
sys.path.insert(0, str(Path(__file__).parent))

from models.clip import CLIPModel, ImageEncoder, TextEncoder, create_model

def test_image_encoder(image_encoder_name='starnet_dual_pyramid_rcf', embed_dim=512):
    """测试图像编码器"""
    print(f"\n{'='*60}")
    print(f"测试图像编码器: {image_encoder_name}")
    print(f"{'='*60}")
    
    try:
        encoder = ImageEncoder(model_name=image_encoder_name, embed_dim=embed_dim)
        print(f"✓ 成功创建图像编码器: {image_encoder_name}")
        
        # 测试前向传播
        dummy_image = torch.randn(2, 3, 224, 224)
        encoder.eval()
        with torch.no_grad():
            features = encoder(dummy_image)
            print(f"✓ 前向传播成功")
            print(f"  输入形状: {dummy_image.shape}")
            print(f"  输出形状: {features.shape}")
            print(f"  特征维度: {features.shape[1]}")
            
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_text_encoder(text_encoder_name='bert-base-chinese', embed_dim=512):
    """测试文本编码器"""
    print(f"\n{'='*60}")
    print(f"测试文本编码器: {text_encoder_name}")
    print(f"{'='*60}")
    
    try:
        encoder = TextEncoder(model_name=text_encoder_name, embed_dim=embed_dim)
        print(f"✓ 成功创建文本编码器: {text_encoder_name}")
        
        # 测试前向传播
        test_texts = ["这是一个测试文本", "这是另一个测试"]
        encoder.eval()
        with torch.no_grad():
            features = encoder(texts=test_texts)
            print(f"✓ 前向传播成功")
            print(f"  输入文本数量: {len(test_texts)}")
            print(f"  输出形状: {features.shape}")
            print(f"  特征维度: {features.shape[1]}")
            
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_clip_model(image_encoder_name='starnet_dual_pyramid_rcf', 
                   text_encoder_name='bert-base-chinese',
                   embed_dim=512):
    """测试完整的CLIP模型"""
    print(f"\n{'='*60}")
    print(f"测试完整CLIP模型")
    print(f"  图像编码器: {image_encoder_name}")
    print(f"  文本编码器: {text_encoder_name}")
    print(f"  嵌入维度: {embed_dim}")
    print(f"{'='*60}")
    
    try:
        model = CLIPModel(
            image_encoder_name=image_encoder_name,
            text_encoder_name=text_encoder_name,
            embed_dim=embed_dim
        )
        print(f"✓ 成功创建CLIP模型")
        
        # 测试前向传播
        dummy_images = torch.randn(2, 3, 224, 224)
        class_texts = ["类别1", "类别2", "类别3"]
        
        model.eval()
        with torch.no_grad():
            image_features, text_features = model(dummy_images, texts=class_texts)
            print(f"✓ 前向传播成功")
            print(f"  图像特征形状: {image_features.shape}")
            print(f"  文本特征形状: {text_features.shape}")
            
            # 计算相似度
            similarity = model.compute_similarity(image_features, text_features)
            print(f"  相似度矩阵形状: {similarity.shape}")
            
            # 测试预测
            predictions, probabilities = model.predict(dummy_images, class_texts)
            print(f"  预测类别: {predictions.tolist()}")
            print(f"  概率形状: {probabilities.shape}")
            
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_encoders():
    """测试多个编码器组合"""
    print(f"\n{'='*60}")
    print(f"测试多个编码器组合")
    print(f"{'='*60}")
    
    # 图像编码器列表
    image_encoders = [
        'starnet_dual_pyramid_rcf',
        'resnet50',
        # 'resnet18',  # 可以添加更多
    ]
    
    # 文本编码器列表
    text_encoders = [
        'bert-base-chinese',
        # 可以添加更多
    ]
    
    embed_dim = 512
    
    results = []
    for img_enc in image_encoders:
        for txt_enc in text_encoders:
            print(f"\n测试组合: {img_enc} + {txt_enc}")
            success = test_clip_model(img_enc, txt_enc, embed_dim)
            results.append((img_enc, txt_enc, success))
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")
    for img_enc, txt_enc, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {img_enc} + {txt_enc}")

if __name__ == "__main__":
    print("="*60)
    print("CLIP模型测试脚本")
    print("="*60)
    
    # 测试图像编码器
    test_image_encoder('starnet_dual_pyramid_rcf')
    
    # 测试文本编码器
    test_text_encoder('bert-base-chinese')
    
    # 测试完整CLIP模型
    test_clip_model()
    
    # 测试多个编码器组合（可选）
    # test_multiple_encoders()
    
    print(f"\n{'='*60}")
    print("测试完成！")
    print(f"{'='*60}")

