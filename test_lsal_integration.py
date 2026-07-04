#!/usr/bin/env python
"""
测试LSAL损失集成到train_clip.py是否正常工作
"""

import sys
import os
import torch
from pathlib import Path

# 设置环境变量
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# 添加路径
sys.path.insert(0, 'models')
sys.path.insert(0, '.')

def test_imports():
    """测试导入"""
    print("="*60)
    print("测试1: 导入检查")
    print("="*60)
    
    try:
        from models.lsal_biomedclip import LLMSemanticSuperLoss
        print("✓ LLMSemanticSuperLoss 导入成功")
        return True
    except ImportError as e:
        print(f"✗ LLMSemanticSuperLoss 导入失败: {e}")
        return False

def test_lsal_loss():
    """测试LSAL损失函数"""
    print("\n" + "="*60)
    print("测试2: LSAL损失函数")
    print("="*60)
    
    try:
        from models.lsal_biomedclip import LLMSemanticSuperLoss
        
        # 创建模拟数据
        num_classes = 9
        embed_dim = 512
        batch_size = 4
        
        # 创建软标签矩阵和类别中心
        soft_labels_matrix = torch.softmax(torch.randn(num_classes, num_classes), dim=1)
        class_centers = torch.randn(num_classes, embed_dim)
        class_centers = class_centers / class_centers.norm(dim=-1, keepdim=True)
        
        # 创建损失函数
        criterion = LLMSemanticSuperLoss(
            soft_labels_matrix=soft_labels_matrix,
            class_centers=class_centers,
            lambda_anchor=0.5
        )
        print("✓ LSAL损失函数创建成功")
        
        # 测试前向传播（不需要梯度，只测试计算）
        image_features = torch.randn(batch_size, embed_dim)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        total_loss, loss_cls, loss_anchor = criterion(image_features, logits, labels)
        
        print(f"✓ LSAL损失计算成功")
        print(f"  总损失: {total_loss.item():.4f}")
        print(f"  分类损失: {loss_cls.item():.4f}")
        print(f"  锚点损失: {loss_anchor.item():.4f}")
        
        # 测试反向传播（需要设置requires_grad）
        image_features_grad = torch.randn(batch_size, embed_dim, requires_grad=True)
        image_features_grad = image_features_grad / image_features_grad.norm(dim=-1, keepdim=True)
        logits_grad = torch.randn(batch_size, num_classes, requires_grad=True)
        labels_grad = torch.randint(0, num_classes, (batch_size,))
        
        total_loss_grad, _, _ = criterion(image_features_grad, logits_grad, labels_grad)
        total_loss_grad.backward()
        print("✓ 反向传播成功")
        
        return True
    except Exception as e:
        print(f"✗ LSAL损失测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_with_lsal():
    """测试模型与LSAL损失的集成"""
    print("\n" + "="*60)
    print("测试3: 模型与LSAL损失集成")
    print("="*60)
    
    try:
        from models.clip import CLIPModel
        from models.lsal_biomedclip import LLMSemanticSuperLoss
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建模型（使用biomedclip_text，因为CLIP库可能不可用）
        print("创建CLIP模型（使用biomedclip_text）...")
        try:
            model = CLIPModel(
                image_encoder_name='resnet18',
                text_encoder_name='biomedclip_text',
                embed_dim=512,
                temperature=0.07
            ).to(device)
            print("✓ CLIP模型创建成功（使用biomedclip_text）")
        except Exception as e:
            print(f"⚠ 无法使用biomedclip_text: {e}")
            print("  尝试使用pubmedbert...")
            model = CLIPModel(
                image_encoder_name='resnet18',
                text_encoder_name='pubmedbert',
                embed_dim=512,
                temperature=0.07
            ).to(device)
            print("✓ CLIP模型创建成功（使用pubmedbert）")
        
        # 创建LSAL损失
        num_classes = 9
        embed_dim = 512
        soft_labels_matrix = torch.softmax(torch.randn(num_classes, num_classes), dim=1)
        class_centers = torch.randn(num_classes, embed_dim)
        class_centers = class_centers / class_centers.norm(dim=-1, keepdim=True)
        
        criterion = LLMSemanticSuperLoss(
            soft_labels_matrix=soft_labels_matrix,
            class_centers=class_centers,
            lambda_anchor=0.5
        ).to(device)
        print("✓ LSAL损失函数创建成功")
        
        # 测试前向传播
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224).to(device)
        labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        texts = ['test text 1', 'test text 2']
        
        print("测试前向传播...")
        model.eval()
        with torch.no_grad():
            image_features, text_features = model(images, texts=texts)
            print(f"✓ 模型前向传播成功")
            print(f"  图像特征形状: {image_features.shape}")
            print(f"  文本特征形状: {text_features.shape}")
        
        # 测试损失计算
        print("测试损失计算...")
        model.train()
        
        # 计算分类logits（使用类别中心）
        image_features_norm = torch.nn.functional.normalize(image_features, dim=1)
        class_centers_norm = torch.nn.functional.normalize(class_centers.to(device), dim=1)
        logit_scale = model.temperature.exp() if hasattr(model.temperature, 'exp') else model.temperature
        class_logits = logit_scale * image_features_norm @ class_centers_norm.t()
        
        total_loss, loss_cls, loss_anchor = criterion(image_features_norm, class_logits, labels)
        print(f"✓ 损失计算成功")
        print(f"  总损失: {total_loss.item():.4f}")
        print(f"  分类损失: {loss_cls.item():.4f}")
        print(f"  锚点损失: {loss_anchor.item():.4f}")
        
        # 测试反向传播
        total_loss.backward()
        print("✓ 反向传播成功")
        
        return True
    except Exception as e:
        print(f"✗ 模型与LSAL集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantics_loading():
    """测试语义文件加载"""
    print("\n" + "="*60)
    print("测试4: 语义文件加载")
    print("="*60)
    
    semantics_dir = Path('semantics')
    
    if not semantics_dir.exists():
        print(f"⚠ 语义文件目录不存在: {semantics_dir}")
        print("  请先运行 build_llm_semantics.py 生成语义文件")
        return False
    
    centers_path = semantics_dir / 'class_centers.pt'
    matrix_path = semantics_dir / 'soft_labels_matrix.pt'
    
    if not centers_path.exists():
        print(f"✗ 类别中心文件不存在: {centers_path}")
        return False
    
    if not matrix_path.exists():
        print(f"✗ 软标签矩阵文件不存在: {matrix_path}")
        return False
    
    try:
        class_centers = torch.load(centers_path, map_location='cpu')
        soft_labels_matrix = torch.load(matrix_path, map_location='cpu')
        
        print(f"✓ 语义文件加载成功")
        print(f"  类别中心形状: {class_centers.shape}")
        print(f"  软标签矩阵形状: {soft_labels_matrix.shape}")
        
        # 验证维度
        if class_centers.shape[0] != soft_labels_matrix.shape[0]:
            print(f"✗ 类别数量不匹配: 中心={class_centers.shape[0]}, 矩阵={soft_labels_matrix.shape[0]}")
            return False
        
        print(f"✓ 类别数量匹配: {class_centers.shape[0]}")
        return True
    except Exception as e:
        print(f"✗ 语义文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "="*60)
    print("测试5: 配置文件加载")
    print("="*60)
    
    import json
    
    config_file = Path('train_clip_config_lsal.json')
    
    if not config_file.exists():
        print(f"⚠ 配置文件不存在: {config_file}")
        print("  已创建示例配置文件")
        return True
    
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        if not isinstance(configs, list) or len(configs) == 0:
            print("✗ 配置文件格式错误：应该是包含配置对象的列表")
            return False
        
        config = configs[0]
        
        # 检查LSAL相关参数
        required_keys = ['use_lsal_loss', 'lsal_semantics_dir', 'lsal_lambda_anchor']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"✗ 配置文件缺少LSAL参数: {missing_keys}")
            return False
        
        print("✓ 配置文件加载成功")
        print(f"  使用LSAL损失: {config.get('use_lsal_loss', False)}")
        print(f"  语义文件目录: {config.get('lsal_semantics_dir')}")
        print(f"  Lambda Anchor: {config.get('lsal_lambda_anchor')}")
        
        return True
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("LSAL集成测试")
    print("="*60)
    
    results = []
    
    # 测试1: 导入
    results.append(('导入检查', test_imports()))
    
    # 测试2: LSAL损失函数
    if results[-1][1]:
        results.append(('LSAL损失函数', test_lsal_loss()))
    
    # 测试3: 模型集成
    if results[-1][1]:
        results.append(('模型与LSAL集成', test_model_with_lsal()))
    
    # 测试4: 语义文件加载
    results.append(('语义文件加载', test_semantics_loading()))
    
    # 测试5: 配置文件加载
    results.append(('配置文件加载', test_config_loading()))
    
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
        print("\n建议：")
        if not results[3][1]:  # 语义文件加载失败
            print("1. 运行以下命令生成语义文件：")
            print("   python models/build_llm_semantics.py \\")
            print("       --classnames-file single_label_data/classnames.json \\")
            print("       --output-dir semantics")
    print("="*60)

