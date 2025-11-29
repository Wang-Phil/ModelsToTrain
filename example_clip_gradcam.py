"""
CLIP模型热力图生成示例脚本
演示如何在Python代码中使用clip_gradcam_visualization模块
"""

import torch
import os
from pathlib import Path
from models.clip import CLIPModel
from clip_gradcam_visualization import visualize_clip_gradcam, CLIPGradCAM, get_target_layer_for_clip

def example_single_image():
    """示例：为单张图像生成热力图"""
    
    # 1. 加载模型
    model_path = "checkpoints/clip_model.pth"  # 替换为你的模型路径
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型
    model = CLIPModel(
        image_encoder_name=checkpoint.get('image_encoder', 'resnet50'),
        text_encoder_name=checkpoint.get('text_encoder', 'bert-base-chinese'),
        embed_dim=checkpoint.get('embed_dim', 512)
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 2. 准备类别文本描述
    class_texts = [
        "正常假体",
        "假体松动",
        "假体周围感染",
        "假体周围骨折",
        "假体磨损"
    ]
    
    # 3. 生成热力图
    image_path = "data/test/image.jpg"  # 替换为你的图像路径
    output_path = "results/gradcam_example.png"
    
    visualize_clip_gradcam(
        model=model,
        image_path=image_path,
        class_texts=class_texts,
        img_size=224,
        save_path=output_path,
        use_cuda=torch.cuda.is_available()
    )
    
    print(f"热力图已保存到: {output_path}")


def example_batch_images():
    """示例：批量处理多张图像"""
    
    # 加载模型（同上）
    model_path = "checkpoints/clip_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = CLIPModel(
        image_encoder_name=checkpoint.get('image_encoder', 'resnet50'),
        text_encoder_name=checkpoint.get('text_encoder', 'bert-base-chinese'),
        embed_dim=checkpoint.get('embed_dim', 512)
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 准备类别文本
    class_texts = [
        "正常假体",
        "假体松动",
        "假体周围感染",
    ]
    
    # 批量处理
    image_dir = Path("data/test")
    output_dir = Path("results/gradcam_batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    for img_path in image_files:
        output_path = output_dir / f"gradcam_{img_path.stem}.png"
        
        try:
            visualize_clip_gradcam(
                model=model,
                image_path=str(img_path),
                class_texts=class_texts,
                img_size=224,
                save_path=str(output_path),
                use_cuda=torch.cuda.is_available()
            )
            print(f"处理完成: {img_path.name} -> {output_path}")
        except Exception as e:
            print(f"处理失败 {img_path.name}: {e}")


def example_custom_target_layer():
    """示例：手动指定目标层"""
    
    # 加载模型
    model_path = "checkpoints/clip_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = CLIPModel(
        image_encoder_name=checkpoint.get('image_encoder', 'resnet50'),
        text_encoder_name=checkpoint.get('text_encoder', 'bert-base-chinese'),
        embed_dim=checkpoint.get('embed_dim', 512)
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 手动指定目标层
    # 例如，对于ResNet，使用layer3而不是layer4
    if hasattr(model.image_encoder.backbone, 'layer3'):
        target_layer = model.image_encoder.backbone.layer3[-1]
    else:
        # 自动查找
        image_encoder_name = checkpoint.get('image_encoder', 'resnet50')
        target_layer = get_target_layer_for_clip(model, image_encoder_name)
    
    # 生成热力图
    class_texts = ["正常", "异常"]
    
    visualize_clip_gradcam(
        model=model,
        image_path="data/test/image.jpg",
        class_texts=class_texts,
        target_layer=target_layer,  # 手动指定目标层
        img_size=224,
        save_path="results/gradcam_custom_layer.png",
        use_cuda=torch.cuda.is_available()
    )


def example_specific_class():
    """示例：可视化特定类别的热力图"""
    
    # 加载模型（同上）
    model_path = "checkpoints/clip_model.pth"
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = CLIPModel(
        image_encoder_name=checkpoint.get('image_encoder', 'resnet50'),
        text_encoder_name=checkpoint.get('text_encoder', 'bert-base-chinese'),
        embed_dim=checkpoint.get('embed_dim', 512)
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 类别文本
    class_texts = [
        "正常假体",
        "假体松动",
        "假体周围感染",
    ]
    
    # 可视化索引为1的类别（"假体松动"）
    visualize_clip_gradcam(
        model=model,
        image_path="data/test/image.jpg",
        class_texts=class_texts,
        target_text_idx=1,  # 指定要可视化的类别索引
        img_size=224,
        save_path="results/gradcam_specific_class.png",
        use_cuda=torch.cuda.is_available()
    )


if __name__ == "__main__":
    print("CLIP模型热力图生成示例")
    print("=" * 60)
    
    # 运行示例（取消注释你想运行的示例）
    
    # 示例1：单张图像
    # example_single_image()
    
    # 示例2：批量处理
    # example_batch_images()
    
    # 示例3：自定义目标层
    # example_custom_target_layer()
    
    # 示例4：特定类别
    # example_specific_class()
    
    print("\n请取消注释上面的示例函数来运行相应功能")

