"""
Enhanced CLIP Model 使用示例

展示如何使用增强版CLIP模型，支持多种损失函数组合
"""

import torch
from clip_enhanced import EnhancedCLIPModel, create_enhanced_model

# ========== 示例1: 基础使用（分类损失 + 对比损失）==========
def example_basic():
    print("=" * 80)
    print("示例1: 基础使用（分类损失 + 对比损失）")
    print("=" * 80)
    
    # 类别文本
    class_texts = [
        "a photo of a normal hip",
        "a photo of a hip prosthesis",
        "a photo of a fractured hip"
    ]
    
    # 创建模型
    model = EnhancedCLIPModel(
        image_encoder_name='resnet50',
        text_encoder_name='pubmedbert',
        embed_dim=512,
        temperature=0.07,
        class_texts=class_texts,
        use_classification_loss=True,
        use_contrastive_loss=True,
        use_sccm_loss=False,
        use_kdsp_loss=False,
        classification_loss_weight=1.0,
        contrastive_loss_weight=1.0
    )
    
    # 模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, len(class_texts), (batch_size,))
    
    # 训练模式
    model.train()
    logits, loss_dict = model(images, labels=labels)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss dict: {loss_dict}")
    print()


# ========== 示例2: 使用SCCM损失（需要类别文本描述）==========
def example_with_sccm():
    print("=" * 80)
    print("示例2: 使用SCCM损失（从JSON文件加载类别描述）")
    print("=" * 80)
    
    # 从JSON文件加载类别文本描述
    class_texts_file = '../class_texts_hip_prosthesis.json'  # 根据实际路径调整
    
    # 创建模型
    model = EnhancedCLIPModel(
        image_encoder_name='resnet50',
        text_encoder_name='biomedclip_text',
        embed_dim=512,
        temperature=0.07,
        class_texts_file=class_texts_file,
        use_classification_loss=True,
        use_contrastive_loss=True,
        use_sccm_loss=True,  # 启用SCCM损失
        use_kdsp_loss=False,
        classification_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        sccm_loss_weight=1.0
    )
    
    # 模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 3, (batch_size,))  # 假设有3个类别
    
    # 训练模式
    model.train()
    logits, loss_dict = model(images, labels=labels)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss dict: {loss_dict}")
    print()


# ========== 示例3: 使用KDSP损失（需要teacher模型）==========
def example_with_kdsp():
    print("=" * 80)
    print("示例3: 使用KDSP损失（知识蒸馏）")
    print("=" * 80)
    
    # 类别文本
    class_texts = [
        "a photo of a normal hip",
        "a photo of a hip prosthesis"
    ]
    
    # 创建teacher模型（通常是更大的模型，例如BiomedCLIP）
    # 这里用相同配置作为示例，实际应该使用更大的模型
    teacher_model = EnhancedCLIPModel(
        image_encoder_name='biomedclip',
        text_encoder_name='biomedclip_text',
        embed_dim=512,
        class_texts=class_texts,
        use_classification_loss=False,  # Teacher不需要计算损失
        use_contrastive_loss=False,
        use_sccm_loss=False,
        use_kdsp_loss=False
    )
    teacher_model.eval()  # 冻结teacher模型
    
    # 创建student模型（通常是更小的模型）
    student_model = EnhancedCLIPModel(
        image_encoder_name='resnet50',
        text_encoder_name='pubmedbert',
        embed_dim=512,
        temperature=0.07,
        class_texts=class_texts,
        teacher_model=teacher_model,  # 传入teacher模型
        use_classification_loss=True,
        use_contrastive_loss=True,
        use_sccm_loss=False,
        use_kdsp_loss=True,  # 启用KDSP损失
        classification_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        kdsp_loss_weight=1.0
    )
    
    # 模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, len(class_texts), (batch_size,))
    
    # 训练模式
    student_model.train()
    logits, loss_dict = student_model(images, labels=labels)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss dict: {loss_dict}")
    print()


# ========== 示例4: 完整组合（所有损失函数）==========
def example_full_combo():
    print("=" * 80)
    print("示例4: 完整组合（所有损失函数）")
    print("=" * 80)
    
    # 从JSON文件加载类别文本描述
    class_texts_file = '../class_texts_hip_prosthesis.json'  # 根据实际路径调整
    
    # 创建teacher模型
    teacher_model = EnhancedCLIPModel(
        image_encoder_name='biomedclip',
        text_encoder_name='biomedclip_text',
        embed_dim=512,
        class_texts_file=class_texts_file,
        use_classification_loss=False,
        use_contrastive_loss=False,
        use_sccm_loss=False,
        use_kdsp_loss=False
    )
    teacher_model.eval()
    
    # 创建student模型（使用所有损失函数）
    model = EnhancedCLIPModel(
        image_encoder_name='resnet50:pmcclip',  # 使用PMC-CLIP预训练的ResNet50
        text_encoder_name='biomedclip_text',
        embed_dim=512,
        temperature=0.07,
        class_texts_file=class_texts_file,
        teacher_model=teacher_model,
        use_classification_loss=True,
        use_contrastive_loss=True,
        use_sccm_loss=True,
        use_kdsp_loss=True,
        classification_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        sccm_loss_weight=1.0,
        kdsp_loss_weight=1.0
    )
    
    # 模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 3, (batch_size,))  # 假设有3个类别
    
    # 训练模式
    model.train()
    logits, loss_dict = model(images, labels=labels)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss dict: {loss_dict}")
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print()


# ========== 示例5: 使用配置字典创建模型 ==========
def example_with_config():
    print("=" * 80)
    print("示例5: 使用配置字典创建模型")
    print("=" * 80)
    
    config = {
        'image_encoder': 'resnet50',
        'text_encoder': 'pubmedbert',
        'embed_dim': 512,
        'temperature': 0.07,
        'class_texts': [
            "a photo of a normal hip",
            "a photo of a hip prosthesis"
        ],
        'use_classification_loss': True,
        'use_contrastive_loss': True,
        'use_sccm_loss': False,
        'use_kdsp_loss': False,
        'classification_loss_weight': 1.0,
        'contrastive_loss_weight': 1.0
    }
    
    model = create_enhanced_model(config)
    
    # 模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 2, (batch_size,))
    
    # 训练模式
    model.train()
    logits, loss_dict = model(images, labels=labels)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Loss dict: {loss_dict}")
    print()


# ========== 示例6: 评估模式（预测）==========
def example_evaluation():
    print("=" * 80)
    print("示例6: 评估模式（预测）")
    print("=" * 80)
    
    class_texts = [
        "a photo of a normal hip",
        "a photo of a hip prosthesis",
        "a photo of a fractured hip"
    ]
    
    # 创建模型
    model = EnhancedCLIPModel(
        image_encoder_name='resnet50',
        text_encoder_name='pubmedbert',
        embed_dim=512,
        class_texts=class_texts
    )
    
    # 模拟数据
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        # 方法1: 使用预编码的类别文本特征
        logits = model(images, labels=None)
        predictions = torch.argmax(logits, dim=1)
        
        # 方法2: 使用predict方法
        predictions2, probabilities = model.predict(images, return_probs=True)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions (method 1): {predictions}")
    print(f"Predictions (method 2): {predictions2}")
    print(f"Probabilities shape: {probabilities.shape}")
    print()


if __name__ == '__main__':
    # 运行示例（根据需要取消注释）
    
    # example_basic()
    # example_with_sccm()
    # example_with_kdsp()
    # example_full_combo()
    # example_with_config()
    # example_evaluation()
    
    print("请取消注释上面的示例函数来运行相应的示例")

