"""
测试CLIP模型热力图生成
"""

import torch
import os
import json
from pathlib import Path
from models.clip import CLIPModel
from clip_gradcam_visualization import visualize_clip_gradcam

def test_gradcam():
    """测试为几张图片生成热力图"""
    
    # 1. 模型配置
    model_path = "checkpoints/clip_models/resnet18_clip:ViT-B/32/fold_1/checkpoint_best.pth"
    image_encoder = "resnet18"
    text_encoder = "clip:ViT-B/32"  # 使用CLIP文本编码器（不依赖transformers）
    embed_dim = 512
    
    # 2. 加载类别文本描述
    with open("class_texts_hip_prosthesis.json", 'r', encoding='utf-8') as f:
        class_texts_dict = json.load(f)
    class_texts = list(class_texts_dict.values())
    class_names = list(class_texts_dict.keys())
    
    print(f"类别数: {len(class_texts)}")
    print(f"类别: {class_names}")
    
    # 3. 加载模型
    print(f"\n加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = CLIPModel(
        image_encoder_name=image_encoder,
        text_encoder_name=text_encoder,
        embed_dim=embed_dim
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("使用GPU")
    else:
        print("使用CPU")
    
    # 4. 选择测试图片
    test_images = [
        "data/test/Native Hip/346_r.jpg",
        "data/test/Good Place/263_l.jpg",
        "data/test/Native Hip/838_l.jpg",
    ]
    
    # 如果文件不存在，随机选择其他图片
    available_images = []
    for img_path in test_images:
        if os.path.exists(img_path):
            available_images.append(img_path)
        else:
            # 从同一目录随机选择一个
            img_dir = os.path.dirname(img_path)
            if os.path.exists(img_dir):
                img_files = list(Path(img_dir).glob("*.jpg"))
                if img_files:
                    available_images.append(str(img_files[0]))
    
    if not available_images:
        # 从test目录随机选择几张图片
        test_dir = Path("data/test")
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                img_files = list(class_dir.glob("*.jpg"))
                if img_files:
                    available_images.append(str(img_files[0]))
                    if len(available_images) >= 3:
                        break
    
    print(f"\n将处理 {len(available_images)} 张图片:")
    for img in available_images:
        print(f"  - {img}")
    
    # 5. 创建输出目录
    output_dir = "clip_gradcam_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. 为每张图片生成热力图
    for i, image_path in enumerate(available_images):
        print(f"\n处理图片 {i+1}/{len(available_images)}: {os.path.basename(image_path)}")
        
        output_name = f"gradcam_{Path(image_path).stem}.png"
        output_path = os.path.join(output_dir, output_name)
        
        try:
            visualize_clip_gradcam(
                model=model,
                image_path=image_path,
                class_texts=class_texts,
                img_size=224,
                save_path=output_path,
                use_cuda=torch.cuda.is_available(),
                image_encoder_name=image_encoder
            )
            print(f"✓ 成功生成热力图: {output_path}")
        except Exception as e:
            print(f"✗ 生成热力图失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    test_gradcam()

