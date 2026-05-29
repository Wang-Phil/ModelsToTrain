#!/usr/bin/env python3
"""
为每个类别生成预测正确的图片的 Grad-CAM 热力图
每个类别最多生成10张
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from train_multiclass import create_model, ImageFolderDataset

# 类别映射
CLASS_TO_IDX = {
    'Acetabular Loosening': 0,
    'Dislocation': 1,
    'Fracture': 2,
    'Good Place': 3,
    'Infection': 4,
    'Native Hip': 5,
    'Spacer': 6,
    'Stem Loosening': 7,
    'Wear': 8
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def load_model(model_name, checkpoint_path, num_classes=9, device='cuda:0'):
    """加载模型"""
    print(f"加载模型: {model_name}")
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 移除 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    model.to(device)
    
    print("✓ 模型加载成功")
    return model


def predict_image(model, image_path, device='cuda:0'):
    """预测单张图片"""
    # 加载和预处理图像
    img = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probs[0, prediction].item()
    
    return prediction, confidence


def letterbox_image(img, target_size=224, fill_color=(0, 0, 0)):
    """
    使用letterbox方式resize图片，保持宽高比，避免变形
    
    Args:
        img: 输入图片 (numpy array, H x W x C)
        target_size: 目标尺寸
        fill_color: padding填充颜色 (RGB)
    
    Returns:
        resized_img: resize后的图片 (target_size x target_size x C)
        scale: 缩放比例
        pad_top, pad_left: padding位置
    """
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    # 计算resize后的尺寸（保持宽高比）
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize图片
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 计算padding
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 添加padding
    if len(img.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=fill_color
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0
        )
    
    return padded, scale, pad_top, pad_left


def remove_letterbox_padding(cam, original_h, original_w, scale, pad_top, pad_left, target_size=224):
    """
    去除letterbox padding，将CAM映射回原始尺寸
    
    Args:
        cam: CAM热力图 (target_size x target_size)
        original_h, original_w: 原始图片尺寸
        scale: 之前的缩放比例
        pad_top, pad_left: padding位置
        target_size: 目标尺寸
    
    Returns:
        cam_original: 映射回原始尺寸的CAM
    """
    # 计算resize后的尺寸
    new_h = int(original_h * scale)
    new_w = int(original_w * scale)
    
    # 去除padding（提取有效区域）
    cam_cropped = cam[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
    
    # Resize回原始尺寸
    cam_original = cv2.resize(cam_cropped, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    
    return cam_original


def generate_gradcam(model, image_path, target_class, device='cuda:0', 
                     alpha=0.4, class_name=None):
    """生成 Grad-CAM 热力图（使用letterbox保持宽高比，避免偏移）"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from grad_cam_starnet import StarNetGradCAM
    
    # 获取目标层
    if hasattr(model, 'stages') and len(model.stages) > 0:
        last_stage = model.stages[-1]
        if len(last_stage) > 1:
            last_block = last_stage[-1]
            if hasattr(last_block, 'dwconv2'):
                target_layers = [last_block.dwconv2]
            else:
                target_layers = [last_block]
        else:
            target_layers = [last_stage[-1]]
    else:
        print("错误: 无法找到目标层")
        return None, None, None
    
    # 创建 Grad-CAM 对象
    gradcam = StarNetGradCAM(model, target_layers, device)
    
    # 加载原始图像
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]
    
    # 保存原始图片用于显示
    img_original = img_rgb.copy()
    img_normalized = img_original.astype(np.float32) / 255.0
    
    # 使用letterbox方式resize，保持宽高比
    img_padded, scale, pad_top, pad_left = letterbox_image(
        img_rgb, target_size=224, fill_color=(0, 0, 0)
    )
    
    # 转换为模型输入格式
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img_padded).unsqueeze(0)
    
    # 生成 CAM（在224x224的padded图片上）
    cam, prediction, confidence = gradcam.generate_cam(input_tensor, target_class)
    
    # 去除padding并映射回原始尺寸
    cam_original = remove_letterbox_padding(
        cam, original_h, original_w, scale, pad_top, pad_left, target_size=224
    )
    
    # 生成热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_original), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # 叠加热力图和原图（使用原始尺寸）
    vis_image = img_normalized * (1 - alpha) + heatmap * alpha
    vis_image = np.clip(vis_image, 0, 1)
    
    return vis_image, prediction, confidence


def main():
    # 配置
    model_name = 'starnet_sa_s1'
    checkpoint_path = 'checkpoints/final_starnet_models/starnet_sa_s1/fold_1/best_model.pth'
    data_dir = 'single_label_data'
    output_dir = 'gradcam_output/correct_predictions_10perclass'
    # 如果GPU内存不足，可以改用CPU或指定其他GPU
    # 尝试使用 GPU 5 (根据之前的日志，GPU 5 可能可用)
    device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
    # 如果遇到CUDA OOM，可以临时改为: device = 'cpu'
    num_classes = 9
    max_per_class = 10
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(model_name, checkpoint_path, num_classes, device)
    
    # 获取所有类别
    classes = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    
    print(f"\n开始处理 {len(classes)} 个类别，每个类别最多 {max_per_class} 张正确预测的图片\n")
    
    total_correct = 0
    total_processed = 0
    
    for class_name in classes:
        if class_name not in CLASS_TO_IDX:
            print(f"跳过未知类别: {class_name}")
            continue
        
        true_label = CLASS_TO_IDX[class_name]
        class_dir = Path(data_dir) / class_name
        
        # 获取所有图片
        img_files = sorted([f for f in class_dir.iterdir() 
                           if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        print(f"\n处理类别: {class_name} (真实标签: {true_label})")
        print(f"  总图片数: {len(img_files)}")
        
        correct_count = 0
        processed_count = 0
        
        for img_file in img_files:
            if correct_count >= max_per_class:
                break
            
            processed_count += 1
            total_processed += 1
            
            # 预测（每次预测后清理缓存）
            prediction, confidence = predict_image(model, img_file, device)
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            
            # 只处理预测正确的
            if prediction == true_label:
                correct_count += 1
                total_correct += 1
                
                print(f"  ✓ [{correct_count}/{max_per_class}] {img_file.name}: "
                      f"预测={prediction} ({IDX_TO_CLASS[prediction]}), "
                      f"置信度={confidence:.2%}")
                
                # 生成热力图（每次生成后清理缓存）
                vis_image, pred, conf = generate_gradcam(
                    model, str(img_file), true_label, device, 
                    class_name=class_name
                )
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                
                if vis_image is not None:
                    # 保存结果（使用原始尺寸）
                    class_name_clean = class_name.replace(' ', '_').replace('/', '_')
                    output_name = f"{class_name_clean}_{img_file.stem}_{model_name}_gradcam.png"
                    output_path = os.path.join(output_dir, output_name)
                    
                    # 获取原始图片尺寸
                    original_h, original_w = vis_image.shape[:2]
                    
                    # 使用PIL直接保存，保持原始尺寸
                    from PIL import Image as PILImage
                    # 将numpy数组转换为PIL Image（值范围0-1，需要转换为0-255）
                    vis_image_uint8 = (vis_image * 255).astype(np.uint8)
                    pil_image = PILImage.fromarray(vis_image_uint8)
                    
                    # 直接保存，保持原始尺寸
                    pil_image.save(output_path, 'PNG', dpi=(150, 150))
        
        print(f"  完成: {correct_count} 张正确预测 / {processed_count} 张已处理")
        
        if correct_count < max_per_class:
            print(f"  注意: 只找到 {correct_count} 张正确预测的图片（目标: {max_per_class}）")
    
    print(f"\n{'='*60}")
    print(f"总结:")
    print(f"  总处理图片数: {total_processed}")
    print(f"  正确预测数: {total_correct}")
    print(f"  准确率: {total_correct/total_processed*100:.2f}%")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

