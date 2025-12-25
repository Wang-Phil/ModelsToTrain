"""
为每个类别生成30张GradCAM热力图
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm

# 导入CLIP模型和GradCAM工具
from models.clip import CLIPModel
from clip_gradcam_visualization import visualize_clip_gradcam, get_target_layer_for_clip


def get_images_by_class(data_dir: str, class_name: str, num_images: int = 30):
    """
    获取指定类别的图片路径
    
    Args:
        data_dir: 数据目录（包含test子目录）
        class_name: 类别名称
        num_images: 需要的图片数量
    
    Returns:
        image_paths: 图片路径列表
    """
    class_dir = Path(data_dir) / "test" / class_name
    
    if not class_dir.exists():
        print(f"警告: 类别目录不存在: {class_dir}")
        return []
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(class_dir.glob(f"*{ext}")))
        image_paths.extend(list(class_dir.glob(f"*{ext.upper()}")))
    
    if len(image_paths) == 0:
        print(f"警告: 类别 {class_name} 没有找到图片")
        return []
    
    # 如果图片数量少于需要的数量，返回所有图片
    if len(image_paths) < num_images:
        print(f"警告: 类别 {class_name} 只有 {len(image_paths)} 张图片，少于需要的 {num_images} 张")
        return [str(p) for p in image_paths]
    
    # 随机选择指定数量的图片
    selected = random.sample(image_paths, num_images)
    return [str(p) for p in selected]


def load_model(checkpoint_path: str, image_encoder: str, text_encoder: str, embed_dim: int = 512):
    """
    加载CLIP模型
    
    Args:
        checkpoint_path: 模型检查点路径
        image_encoder: 图像编码器名称
        text_encoder: 文本编码器名称
        embed_dim: 嵌入维度
    
    Returns:
        model: 加载的模型
    """
    print(f"\n加载模型: {checkpoint_path}")
    
    # 创建模型
    model = CLIPModel(
        image_encoder_name=image_encoder,
        text_encoder_name=text_encoder,
        embed_dim=embed_dim
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU")
    else:
        print("使用CPU")
    
    return model


def generate_gradcam_for_class(
    model: CLIPModel,
    class_name: str,
    class_text: str,
    class_texts: List[str],
    class_idx: int,
    image_paths: List[str],
    output_dir: str,
    image_encoder_name: str,
    img_size: int = 224
):
    """
    为指定类别的所有图片生成GradCAM热力图
    
    Args:
        model: CLIP模型
        class_name: 类别名称
        class_text: 该类别的文本描述
        class_texts: 所有类别的文本描述列表
        class_idx: 类别索引
        image_paths: 该类别的图片路径列表
        output_dir: 输出目录
        image_encoder_name: 图像编码器名称
        img_size: 图像大小
    """
    # 创建类别输出目录
    class_output_dir = Path(output_dir) / class_name
    class_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n处理类别: {class_name}")
    print(f"  类别文本: {class_text}")
    print(f"  图片数量: {len(image_paths)}")
    print(f"  输出目录: {class_output_dir}")
    
    # 获取目标层
    target_layer = get_target_layer_for_clip(model, image_encoder_name)
    
    # 为每张图片生成热力图
    success_count = 0
    for i, image_path in enumerate(tqdm(image_paths, desc=f"生成 {class_name} 热力图")):
        try:
            # 生成文件名
            image_name = Path(image_path).stem
            output_name = f"{image_name}_gradcam.png"
            output_path = class_output_dir / output_name
            
            # 生成热力图（针对当前类别）
            # visualize_clip_gradcam会自动保存完整可视化图和overlay图
            visualize_clip_gradcam(
                model=model,
                image_path=image_path,
                class_texts=class_texts,
                target_layer=target_layer,
                target_text_idx=class_idx,  # 针对当前类别生成热力图
                img_size=img_size,
                save_path=str(output_path),
                use_cuda=torch.cuda.is_available(),
                image_encoder_name=image_encoder_name
            )
            
            success_count += 1
            
        except Exception as e:
            print(f"\n错误: 处理图片 {image_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"✓ 成功生成 {success_count}/{len(image_paths)} 张热力图")


def main():
    """主函数"""
    # 配置参数
    checkpoint_path = "checkpoints/clip_models/resnet18_clip_ViT-B_32/checkpoint_best.pth"
    image_encoder = "resnet18"
    text_encoder = "clip:ViT-B/32"
    embed_dim = 512
    img_size = 224
    num_images_per_class = 30
    data_dir = "data"
    output_dir = "clip_gradcam_results_by_class"
    
    # 加载类别文本描述
    class_texts_file = "class_texts_hip_prosthesis.json"
    if not os.path.exists(class_texts_file):
        print(f"错误: 找不到类别文本文件: {class_texts_file}")
        return
    
    with open(class_texts_file, 'r', encoding='utf-8') as f:
        class_texts_dict = json.load(f)
    
    class_names = list(class_texts_dict.keys())
    class_texts = list(class_texts_dict.values())
    
    print(f"类别数量: {len(class_names)}")
    print(f"类别列表: {class_names}")
    
    # 加载模型
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到模型文件: {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, image_encoder, text_encoder, embed_dim)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个类别生成热力图
    for class_idx, class_name in enumerate(class_names):
        # 获取该类别的图片
        image_paths = get_images_by_class(data_dir, class_name, num_images_per_class)
        
        if len(image_paths) == 0:
            print(f"\n跳过类别 {class_name}: 没有找到图片")
            continue
        
        # 生成热力图
        generate_gradcam_for_class(
            model=model,
            class_name=class_name,
            class_text=class_texts_dict[class_name],
            class_texts=class_texts,
            class_idx=class_idx,
            image_paths=image_paths,
            output_dir=output_dir,
            image_encoder_name=image_encoder,
            img_size=img_size
        )
    
    print(f"\n✓ 所有热力图生成完成！")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性（可选）
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    main()

