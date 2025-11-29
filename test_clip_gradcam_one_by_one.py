"""
逐张生成CLIP热力图 - 每个类别最多10张预测正确的图片
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
from pathlib import Path
from collections import defaultdict
from torchvision import transforms
from PIL import Image

print("正在加载模块...")

try:
    from models.clip import CLIPModel
    from clip_gradcam_visualization import visualize_clip_gradcam
    print("✓ 模块加载成功")
except ImportError as e:
    print(f"✗ 导入错误: {e}")
    print("\n解决方案:")
    print("1. 降级NumPy: pip install 'numpy<2.0'")
    print("2. 安装CLIP库: pip install git+https://github.com/openai/CLIP.git")
    exit(1)
except Exception as e:
    print(f"✗ 其他错误: {type(e).__name__}: {e}")
    exit(1)

def main():
    """主函数：逐张处理图片，每个类别生成最多10张"""
    
    # 配置参数
    model_path = "checkpoints/clip_models/resnet18_clip:ViT-B/32/fold_1/checkpoint_best.pth"
    image_encoder = "resnet18"
    text_encoder = "bert-base-chinese"
    embed_dim = 512
    max_images_per_class = 10
    test_dir = "data/test"
    output_dir = "clip_gradcam_results_by_class"
    
    # 1. 加载类别文本
    class_texts_file = "class_texts_hip_prosthesis.json"
    if not os.path.exists(class_texts_file):
        print(f"✗ 类别文本文件不存在: {class_texts_file}")
        return
    
    with open(class_texts_file, 'r', encoding='utf-8') as f:
        class_texts_dict = json.load(f)
    class_texts = list(class_texts_dict.values())
    class_names = list(class_texts_dict.keys())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\n类别数: {len(class_names)}")
    print(f"类别: {class_names}\n")
    
    # 2. 加载模型
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = CLIPModel(
        image_encoder_name=image_encoder,
        text_encoder_name=text_encoder,
        embed_dim=embed_dim
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ 完整模型权重加载成功")
    except RuntimeError:
        print("⚠ 完整加载失败，尝试只加载图像编码器...")
        image_encoder_dict = {k.replace('image_encoder.', ''): v 
                             for k, v in state_dict.items() 
                             if k.startswith('image_encoder.')}
        if image_encoder_dict:
            model.image_encoder.load_state_dict(image_encoder_dict, strict=False)
            print(f"✓ 图像编码器权重加载成功（{len(image_encoder_dict)}个参数）")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"✓ 使用设备: {device}\n")
    
    # 3. 准备数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    for class_name in class_names:
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)
    
    # 5. 收集所有图片
    test_path = Path(test_dir)
    images_by_class = defaultdict(list)
    for class_dir in sorted(test_path.iterdir()):
        if class_dir.is_dir() and class_dir.name in class_to_idx:
            img_files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
            images_by_class[class_dir.name] = img_files
            print(f"  类别 '{class_dir.name}': {len(img_files)} 张图片")
    
    print(f"\n{'='*70}")
    print(f"开始逐张处理（每个类别最多生成{max_images_per_class}张预测正确的图片）")
    print(f"{'='*70}\n")
    
    # 6. 逐张处理
    stats_by_class = defaultdict(lambda: {'processed': 0, 'correct': 0, 'generated': 0})
    total_processed = 0
    total_correct = 0
    
    for class_name in sorted(class_names):
        if class_name not in images_by_class:
            continue
        
        print(f"\n处理类别: {class_name}")
        print("-" * 70)
        
        images = images_by_class[class_name]
        true_label_idx = class_to_idx[class_name]
        generated_count = 0
        
        for img_path in images:
            if generated_count >= max_images_per_class:
                break
            
            try:
                # 加载图片
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                
                # 预测
                with torch.no_grad():
                    predictions, probabilities = model.predict(input_tensor, class_texts)
                    pred_idx = predictions[0].item()
                    pred_prob = probabilities[0][pred_idx].item()
                    pred_class = class_names[pred_idx]
                
                stats_by_class[class_name]['processed'] += 1
                total_processed += 1
                
                # 只处理预测正确的图片
                if pred_idx == true_label_idx:
                    stats_by_class[class_name]['correct'] += 1
                    total_correct += 1
                    
                    # 生成热力图
                    output_name = f"gradcam_{img_path.stem}.png"
                    output_path = os.path.join(output_dir, class_name, output_name)
                    
                    try:
                        visualize_clip_gradcam(
                            model=model,
                            image_path=str(img_path),
                            class_texts=class_texts,
                            img_size=224,
                            save_path=output_path,
                            use_cuda=torch.cuda.is_available(),
                            image_encoder_name=image_encoder
                        )
                        generated_count += 1
                        stats_by_class[class_name]['generated'] += 1
                        print(f"  ✓ [{generated_count}/{max_images_per_class}] {img_path.name}")
                        print(f"    预测: {pred_class}, 概率: {pred_prob:.3f}")
                    except Exception as e:
                        print(f"  ✗ 生成热力图失败 {img_path.name}: {e}")
                else:
                    # 预测错误，跳过
                    pass
                    
            except Exception as e:
                print(f"  ✗ 处理失败 {img_path.name}: {e}")
        
        # 打印该类别的统计
        stats = stats_by_class[class_name]
        if stats['processed'] > 0:
            acc = stats['correct'] / stats['processed'] * 100
            print(f"\n  类别统计: 处理 {stats['processed']} 张, "
                  f"正确 {stats['correct']} 张 ({acc:.1f}%), "
                  f"生成 {stats['generated']} 张热力图")
    
    # 7. 打印最终统计
    print(f"\n{'='*70}")
    print("处理完成！")
    print(f"{'='*70}")
    print(f"\n总计:")
    print(f"  处理图片数: {total_processed}")
    print(f"  预测正确数: {total_correct}")
    if total_processed > 0:
        print(f"  总体准确率: {total_correct/total_processed*100:.2f}%")
    
    print(f"\n按类别统计:")
    for class_name in sorted(class_names):
        if class_name in stats_by_class:
            stats = stats_by_class[class_name]
            if stats['processed'] > 0:
                acc = stats['correct'] / stats['processed'] * 100
                print(f"  {class_name}:")
                print(f"    处理: {stats['processed']}, 正确: {stats['correct']} ({acc:.1f}%), "
                      f"生成: {stats['generated']}")
    
    total_generated = sum(s['generated'] for s in stats_by_class.values())
    print(f"\n总共生成: {total_generated} 张热力图")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    print("="*70)
    print("CLIP模型热力图生成 - 逐张处理")
    print("="*70)
    main()

