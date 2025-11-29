"""
简化版CLIP热力图测试 - 尝试绕过环境问题
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
from pathlib import Path

print("正在加载模块...")

# 尝试导入，如果失败则给出明确提示
try:
    from models.clip import CLIPModel
    from clip_gradcam_visualization import visualize_clip_gradcam
    print("✓ 模块加载成功")
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("\n解决方案:")
    print("1. 降级NumPy: pip install 'numpy<2.0'")
    print("2. 安装CLIP库: pip install git+https://github.com/openai/CLIP.git")
    print("3. 或者使用conda环境重新安装依赖")
    exit(1)
except Exception as e:
    print(f"✗ 其他错误: {type(e).__name__}: {e}")
    print("\n这可能是NumPy版本冲突。尝试:")
    print("pip install 'numpy<2.0' 'scipy<1.12'")
    exit(1)

def test_gradcam():
    """测试为几张图片生成热力图"""
    
    # 1. 模型配置 - 使用CLIP文本编码器（匹配checkpoint）
    model_path = "checkpoints/clip_models/resnet18_clip:ViT-B/32/fold_1/checkpoint_best.pth"
    image_encoder = "resnet18"
    # 尝试使用CLIP，如果失败则使用BERT
    text_encoder = "bert-base-chinese"  # 使用BERT文本编码器（CLIP未安装时）
    embed_dim = 512
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"✗ 模型文件不存在: {model_path}")
        print("\n可用的模型:")
        import glob
        models = glob.glob("checkpoints/clip_models/**/*.pth", recursive=True)
        for m in models[:5]:
            print(f"  - {m}")
        return
    
    # 2. 加载类别文本描述
    class_texts_file = "class_texts_hip_prosthesis.json"
    if not os.path.exists(class_texts_file):
        print(f"✗ 类别文本文件不存在: {class_texts_file}")
        return
    
    with open(class_texts_file, 'r', encoding='utf-8') as f:
        class_texts_dict = json.load(f)
    class_texts = list(class_texts_dict.values())
    class_names = list(class_texts_dict.keys())
    
    print(f"\n类别数: {len(class_texts)}")
    print(f"类别: {class_names[:3]}...")
    
    # 3. 加载模型
    print(f"\n加载模型: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        model = CLIPModel(
            image_encoder_name=image_encoder,
            text_encoder_name=text_encoder,
            embed_dim=embed_dim
        )
        
        # 加载权重（只加载匹配的部分）
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 尝试完整加载，如果失败则只加载图像编码器
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ 完整模型权重加载成功")
        except RuntimeError as e:
            print("⚠ 完整加载失败，尝试只加载图像编码器...")
            # 只加载图像编码器的权重
            image_encoder_dict = {k.replace('image_encoder.', ''): v 
                                  for k, v in state_dict.items() 
                                  if k.startswith('image_encoder.')}
            if image_encoder_dict:
                model.image_encoder.load_state_dict(image_encoder_dict, strict=False)
                print(f"✓ 图像编码器权重加载成功（{len(image_encoder_dict)}个参数）")
            else:
                print("⚠ 未找到图像编码器权重")
        
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
            print("✓ 使用GPU")
        else:
            print("✓ 使用CPU")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 选择测试图片
    test_images = [
        "data/test/Native Hip/346_r.jpg",
        "data/test/Good Place/263_l.jpg",
        "data/test/Acetabular Loosening/1012_r.jpg",
    ]
    
    # 如果文件不存在，从目录中查找
    available_images = []
    for img_path in test_images:
        if os.path.exists(img_path):
            available_images.append(img_path)
        else:
            # 从同一目录选择第一张图片
            img_dir = os.path.dirname(img_path)
            if os.path.exists(img_dir):
                img_files = sorted(Path(img_dir).glob("*.jpg"))
                if img_files:
                    available_images.append(str(img_files[0]))
    
    if not available_images:
        # 从test目录选择
        test_dir = Path("data/test")
        for class_dir in sorted(test_dir.iterdir()):
            if class_dir.is_dir():
                img_files = sorted(class_dir.glob("*.jpg"))
                if img_files:
                    available_images.append(str(img_files[0]))
                    if len(available_images) >= 3:
                        break
    
    print(f"\n将处理 {len(available_images)} 张图片:")
    for img in available_images:
        print(f"  - {img}")
    
    if not available_images:
        print("✗ 没有找到测试图片")
        return
    
    # 5. 创建输出目录
    output_dir = "clip_gradcam_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 6. 为每张图片生成热力图
    success_count = 0
    for i, image_path in enumerate(available_images):
        print(f"\n{'='*60}")
        print(f"处理图片 {i+1}/{len(available_images)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
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
            success_count += 1
        except Exception as e:
            print(f"✗ 生成热力图失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"完成！成功处理 {success_count}/{len(available_images)} 张图片")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("CLIP模型热力图生成测试（简化版）")
    print("="*60)
    test_gradcam()


